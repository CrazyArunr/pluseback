"""
MongoDB API for test analysis and reporting
"""

import google.generativeai as genai
import json
from pymongo import MongoClient
from bson import ObjectId, json_util
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
from pymongo.errors import ConnectionFailure, PyMongoError
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import hashlib
import re
import time
from io import BytesIO
import zipfile

from ..config.settings import (
    GEMINI_API_KEY,
    MONGO_URI,
    DB_NAME,
    COLLECTION_NAME,
    CONNECT_TIMEOUT_MS,
    SOCKET_TIMEOUT_MS
)

# Create router
router = APIRouter(
    tags=["MongoDB API"],
    responses={404: {"description": "Not found"}},
)

# Load environment variables
load_dotenv()

# Pydantic models for request/response
class QueryModel(BaseModel):
    query: str

# New models for report analysis
class ReportAnalysisRequest(BaseModel):
    report_id: str
    analyze_with_ai: Optional[bool] = False

class ReportAnalysisResponse(BaseModel):
    report_id: str
    status: str
    environment: str
    test_count: int
    date: str
    test_details: list
    raw_document: dict
    ai_analysis: Optional[str] = None

class AIClassificationResponse(BaseModel):
    classification: str
    reason: str

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config={
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 1000,
    },
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
)

# New helper functions for report analysis
def serialize_document(document: dict) -> dict:
    """Convert MongoDB document to JSON-serializable format"""
    return json.loads(json_util.dumps(document))

def get_report_document(report_id: str):
    """Fetch a report document from MongoDB"""
    try:
        obj_id = ObjectId(report_id)
        document = MongoDBManager.get_client()[DB_NAME][COLLECTION_NAME].find_one({"_id": obj_id})
        if not document:
            raise HTTPException(status_code=404, detail="Report not found")
        return document
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid report ID: {str(e)}")

def extract_failed_steps(document: dict):
    """Extract only failed/error/skipped steps from the test document"""
    failed_steps = []
    
    # Function to recursively search for test data and failures
    def find_failures_recursive(obj, path=""):
        nonlocal failed_steps
        
        if isinstance(obj, dict):
            # Check if this object represents a failed test/step
            status_fields = ["status", "result", "state", "outcome"]
            status_value = None
            
            for status_field in status_fields:
                if status_field in obj:
                    status_value = str(obj[status_field]).lower()
                    break
            
            # Check for failure indicators
            if status_value and any(indicator in status_value for indicator in ["fail", "error", "skip", "abort", "block"]):
                failed_step = {
                    "step_name": obj.get("name", obj.get("testName", obj.get("title", obj.get("description", f"Step at {path}")))),
                    "status": status_value,
                    "error_message": obj.get("error", obj.get("errorMessage", obj.get("message", obj.get("reason", obj.get("details", ""))))),
                    "step_number": obj.get("step", obj.get("stepNumber", obj.get("index", ""))),
                    "expected": obj.get("expected", obj.get("expectedResult", "")),
                    "actual": obj.get("actual", obj.get("actualResult", "")),
                    "screenshot": obj.get("screenshot", obj.get("screenshotPath", "")),
                    "duration": obj.get("duration", obj.get("time", obj.get("executionTime", ""))),
                    "path": path
                }
                failed_steps.append(failed_step)
            
            # Also check for error fields even without explicit status
            error_fields = ["error", "errorMessage", "exception", "failure"]
            has_error = any(field in obj and obj[field] for field in error_fields)
            
            if has_error and not any(step.get("path") == path for step in failed_steps):
                failed_step = {
                    "step_name": obj.get("name", obj.get("testName", obj.get("title", f"Error at {path}"))),
                    "status": "error",
                    "error_message": obj.get("error", obj.get("errorMessage", obj.get("exception", obj.get("failure", "")))),
                    "step_number": obj.get("step", obj.get("stepNumber", "")),
                    "expected": obj.get("expected", ""),
                    "actual": obj.get("actual", ""),
                    "path": path
                }
                failed_steps.append(failed_step)
            
            # Continue searching in nested objects
            for key, value in obj.items():
                if key not in ["_id", "timestamp", "createdAt", "updatedAt"]:  # Skip metadata
                    find_failures_recursive(value, f"{path}.{key}" if path else key)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                find_failures_recursive(item, f"{path}[{i}]" if path else f"[{i}]")
    
    # Start recursive search
    find_failures_recursive(document)
    
    # If no failures found with recursive search, try common field names
    if not failed_steps:
        common_test_fields = ["tests", "testCases", "results", "steps", "scenarios", "testResults", "executions"]
        
        for field in common_test_fields:
            if field in document and isinstance(document[field], list):
                for i, test in enumerate(document[field]):
                    if isinstance(test, dict):
                        # Check overall test status
                        overall_status = document.get("status", "").lower()
                        if "fail" in overall_status or "error" in overall_status:
                            failed_step = {
                                "step_name": test.get("name", f"Test {i+1}"),
                                "status": overall_status,
                                "error_message": str(test),  # Include full test data for AI analysis
                                "step_number": i+1,
                                "path": f"{field}[{i}]"
                            }
                            failed_steps.append(failed_step)
    
    return failed_steps

def analyze_with_ai(document: dict):
    """Analyze only failed steps using Gemini AI for precise classification"""
    try:
        # Extract only failed steps
        failed_steps = extract_failed_steps(document)
        
        # Also check document-level status
        doc_status = document.get("status", "").lower()
        overall_failed = any(indicator in doc_status for indicator in ["fail", "error", "abort", "block"])
        
        if not failed_steps and not overall_failed:
            return {
                "classification": "Passed",
                "reason": "No failed steps found in the test execution."
            }
        
        # If no specific failed steps but overall failure, use document summary
        analysis_data = failed_steps if failed_steps else {
            "document_status": doc_status,
            "document_summary": {k: v for k, v in document.items() if k not in ["_id"] and not k.startswith("_")},
            "note": "No specific failed steps found, analyzing overall document"
        }
        
        # Focused prompt for failure-based classification
        prompt = f"""Analyze the test failure data below and classify the test type based on failure patterns and scope.

Failure Data:
{json.dumps(analysis_data, indent=2, default=str)}

Classification Rules:
- **Smoke**: Basic critical functionality failures (login, main page load, core features, app startup issues)
- **Sanity**: Narrow-focused failures after minor changes, specific feature verification, limited scope
- **Regression**: Multiple failures across different modules, existing functionality broken, system-wide issues

Analysis Criteria:
1. **Smoke Testing Indicators**:
   - Login/authentication failures
   - Main application startup issues  
   - Core navigation failures
   - Critical path blockages
   - Basic functionality not working

2. **Sanity Testing Indicators**:
   - Specific feature/module failures
   - Limited scope failures after recent changes
   - Single workflow verification failures
   - Focused area testing

3. **Regression Testing Indicators**:
   - Multiple unrelated failures
   - Previously working features now broken
   - Cross-module failure patterns
   - System-wide functionality issues

Based on the failure data above:
- If critical basic functionality failed → Smoke
- If specific focused area failed → Sanity  
- If multiple areas or existing features failed → Regression

Provide ONLY this format:
Classification: [Smoke/Sanity/Regression]
Reason: [Single sentence explaining why this classification based on failure scope and impact.]

No other text or formatting."""
        
        response = model.generate_content(prompt)
        ai_text = response.text.strip()
        
        # Parse the AI response
        lines = [line.strip() for line in ai_text.split('\n') if line.strip()]
        classification = ""
        reason = ""
        
        for line in lines:
            if line.startswith("Classification:"):
                classification = line.replace("Classification:", "").strip()
            elif line.startswith("Reason:"):
                reason = line.replace("Reason:", "").strip()
                break  # Only take the first reason line
        
        return {
            "classification": classification,
            "reason": reason.strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

def get_failure_summary(document: dict):
    """Extract failure summary from test document"""
    try:
        failures = []
        
        # Check for test details in different possible locations
        test_data = None
        if "tests" in document and isinstance(document["tests"], list):
            test_data = document["tests"]
        elif "testCases" in document and isinstance(document["testCases"], list):
            test_data = document["testCases"]
        elif "results" in document and isinstance(document["results"], list):
            test_data = document["results"]
        
        if test_data:
            for test in test_data:
                if isinstance(test, dict):
                    status = test.get("status", "").lower()
                    if status in ["failed", "error", "skipped"]:
                        failure_info = {
                            "test_name": test.get("name", test.get("testName", "Unknown")),
                            "status": status,
                            "error": test.get("error", test.get("errorMessage", "")),
                            "step": test.get("step", test.get("stepNumber", "")),
                        }
                        failures.append(failure_info)
        
        return failures
    except Exception as e:
        return []

# New endpoints for report analysis
@router.post("/analyze-report")
async def analyze_report(request: ReportAnalysisRequest):
    """
    Analyze a test report with optional AI analysis
    
    Parameters:
    - report_id: The MongoDB document ID
    - analyze_with_ai: Whether to include Gemini AI analysis (default: False)
    
    Returns:
    - If analyze_with_ai=True: Classification, reason, and detailed failure analysis
    - If analyze_with_ai=False: Full report details
    """
    try:
        # Get the raw document
        document = get_report_document(request.report_id)
        
        # If AI analysis is requested, return classification based on failed steps only
        if request.analyze_with_ai:
            ai_result = analyze_with_ai(document)
            
            return AIClassificationResponse(
                classification=ai_result["classification"],
                reason=ai_result["reason"]
            )
        
        # Otherwise, return full report details (original behavior)
        serialized_doc = serialize_document(document)
        
        # Prepare response data
        response_data = {
            "report_id": str(document["_id"]),
            "status": document.get("status", "Unknown"),
            "environment": document.get("environment", "Unspecified"),
            "date": str(document.get("timestamp", "No date available")),
            "test_details": [],
            "raw_document": serialized_doc,
        }
        
        # Handle test details
        if "tests" in document and isinstance(document["tests"], list):
            response_data["test_details"] = document["tests"]
        elif "testCases" in document and isinstance(document["testCases"], list):
            response_data["test_details"] = document["testCases"]
        
        response_data["test_count"] = len(response_data["test_details"])
        
        return ReportAnalysisResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/debug-document/{report_id}")
async def debug_document_structure(report_id: str):
    """
    Debug endpoint to see the actual document structure and help identify failures
    """
    try:
        document = get_report_document(report_id)
        failed_steps = extract_failed_steps(document)
        
        return {
            "report_id": report_id,
            "document_keys": list(document.keys()),
            "document_status": document.get("status", "No status field"),
            "failed_steps_found": len(failed_steps),
            "failed_steps": failed_steps,
            "full_document": serialize_document(document)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

class MongoDBManager:
    @staticmethod
    def get_client() -> MongoClient:
        """Create and return a secure MongoDB client"""
        try:
            client = MongoClient(
                MONGO_URI,
                connectTimeoutMS=CONNECT_TIMEOUT_MS,
                socketTimeoutMS=SOCKET_TIMEOUT_MS,
                retryWrites=True,
                serverSelectionTimeoutMS=5000,
                tls=True,
                tlsAllowInvalidCertificates=True 
            )
            client.admin.command('ping')
            print("✅ Successfully connected to MongoDB")
            return client
        except ConnectionFailure as e:
            raise ConnectionError(f"MongoDB connection failed: {e}") from e

    @staticmethod
    def get_test_data_by_project_ids(project_ids: List[str], days_back: Optional[int] = None) -> List[Dict]:
        """Fetch test data for exactly two projects"""
        if len(project_ids) != 2:
            raise ValueError("Exactly two project IDs must be provided for comparison")
            
        client = None
        try:
            client = MongoDBManager.get_client()
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]
            
            # Build query with both _id and projectId options
            query = {
                "_class": "pulse_qa_api.entity.GenerateReport",
                "$or": [
                    {"_id": ObjectId(pid) if ObjectId.is_valid(pid) else None} for pid in project_ids
                ] + [
                    {"projectId": pid} for pid in project_ids
                ]
            }
            
            # Clean up query by removing None conditions
            query["$or"] = [cond for cond in query["$or"] if list(cond.values())[0] is not None]
            
            if days_back:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
                query["scenarios.start_timestamp"] = {"$gte": cutoff_date.isoformat()}
            
            docs = list(collection.find(query))
            
            if len(docs) != 2:
                raise ValueError(f"Could not find both projects. Found {len(docs)}/2")
            
            # Convert ObjectIds to strings
            for doc in docs:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                if 'projectId' in doc and isinstance(doc['projectId'], ObjectId):
                    doc['projectId'] = str(doc['projectId'])
            
            # Ensure we return exactly two projects in consistent order
            ordered_docs = []
            for pid in project_ids:
                found = next((d for d in docs if d.get('_id') == pid or d.get('projectId') == pid), None)
                if found:
                    ordered_docs.append(found)
            
            return ordered_docs
            
        except PyMongoError as e:
            raise RuntimeError(f"Database operation failed: {str(e)}") from e
        finally:
            if client:
                client.close()

class ProjectComparator:
    @staticmethod
    def generate_tabular_comparison(projects_data: List[Dict]) -> str:
        """Generate markdown table comparing scenarios across two projects"""
        project1_name = projects_data[0].get('projectName', 'Project 1')
        project2_name = projects_data[1].get('projectName', 'Project 2')
        
        all_steps = set()
        for project in projects_data:
            for scenario in project.get('scenarios', []):
                for step in scenario.get('steps', []):
                    all_steps.add(step['name'])
        
        table = [
            f"| Step | {project1_name} Status | {project2_name} Status | Observation |",
            "|------|----------------------|----------------------|-------------|"
        ]
        
        for step in sorted(all_steps):
            row = [f"| {step} "]
            observations = []
            statuses = {}
            
            for i, project in enumerate(projects_data, 1):
                found = False
                for scenario in project.get('scenarios', []):
                    for s in scenario.get('steps', []):
                        if s['name'] == step:
                            status = "✅ Passed" if s['result']['status'] == 'passed' else "❌ Failed"
                            statuses[f"project{i}"] = status
                            found = True
                            break
                    if found:
                        break
                if not found:
                    statuses[f"project{i}"] = "–"
            
            row.append(f"| {statuses['project1']} ")
            row.append(f"| {statuses['project2']} ")
            
            # Generate observations
            if statuses['project1'] == statuses['project2']:
                if statuses['project1'] == "✅ Passed":
                    observations.append("Stable across both")
                elif statuses['project1'] == "❌ Failed":
                    observations.append("Consistent failure")
            else:
                if "browser" in step.lower():
                    observations.append("Browser initialization issue")
                elif "navigate" in step.lower():
                    observations.append("Page load problem")
                elif "click" in step.lower():
                    observations.append("Element interaction issue")
                elif "enter" in step.lower():
                    observations.append("Input field problem")
                elif "wait" in step.lower():
                    observations.append("Timing issue")
                else:
                    observations.append("Behavior difference")
            
            row.append(f"| {' '.join(observations)} |")
            table.append(''.join(row))
        
        return "\n".join(table)

    @staticmethod
    def compare_projects(projects_data: List[Dict]) -> Dict:
        """Compare test data across two projects"""
        comparison = {
            "project_names": [],
            "total_scenarios": 0,
            "total_failures": 0,
            "failure_rates": {},
            "common_failures": [],
            "scenario_comparison": {},
            "tabular_comparison": ProjectComparator.generate_tabular_comparison(projects_data)
        }
        
        # ... rest of the existing ProjectComparator class implementation ...

@router.get("/compare-reports")
async def compare_reports(
    report_ids: List[str] = Query(..., description="Exactly two report IDs to compare"),
    days_back: int = 7
):
    """Main endpoint for comparing two reports using their _id fields"""
    try:
        if len(report_ids) != 2:
            raise HTTPException(
                status_code=400,
                detail="Exactly two report IDs must be provided for comparison"
            )

        client = None
        try:
            client = MongoDBManager.get_client()
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]

            # Build query to handle both ObjectId and UUID-style IDs
            query = {
                "_class": "pulseQA_Web.entity.GenerateReport",
                "$or": [
                    {"_id": ObjectId(rid) if ObjectId.is_valid(rid) else None} for rid in report_ids
                ] + [
                    {"reportId": rid} for rid in report_ids
                ]
            }
            
            # Clean up query by removing None conditions
            query["$or"] = [cond for cond in query["$or"] if list(cond.values())[0] is not None]

            if days_back:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
                query["scenarios.start_timestamp"] = {"$gte": cutoff_date.isoformat()}

            # Add debug logging
            print(f"Searching for reports with query: {query}")
            
            docs = list(collection.find(query))
            print(f"Found {len(docs)} documents")

            # Verify we found exactly two reports
            found_ids = {str(doc.get('_id', '')) for doc in docs}
            found_report_ids = {str(doc.get('reportId', '')) for doc in docs}
            
            # Check which IDs were not found
            missing_ids = []
            for rid in report_ids:
                if rid not in found_ids and rid not in found_report_ids:
                    missing_ids.append(rid)

            if missing_ids:
                # Try to find any documents with these IDs to help debug
                debug_query = {
                    "$or": [
                        {"_id": ObjectId(rid) if ObjectId.is_valid(rid) else None} for rid in missing_ids
                    ] + [
                        {"reportId": rid} for rid in missing_ids
                    ]
                }
                debug_docs = list(collection.find(debug_query))
                print(f"Debug search found {len(debug_docs)} documents")
                
                if debug_docs:
                    print("Found documents with similar IDs:")
                    for doc in debug_docs:
                        print(f"Doc ID: {doc.get('_id')}, Report ID: {doc.get('reportId')}")

                raise HTTPException(
                    status_code=404,
                    detail=f"Could not find reports with IDs: {missing_ids}"
                )

            # Convert ObjectId to string for response
            for doc in docs:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                if 'reportId' in doc and isinstance(doc['reportId'], ObjectId):
                    doc['reportId'] = str(doc['reportId'])

            # Ensure order matches request order
            ordered_docs = []
            for rid in report_ids:
                doc = next((d for d in docs if d.get('_id') == rid or d.get('reportId') == rid), None)
                if doc:
                    ordered_docs.append(doc)

            # Get report names and IDs early
            report_names = [doc.get('reportName', '') for i, doc in enumerate(ordered_docs)]
            report_ids = [doc.get('_id') for doc in ordered_docs]

            # Calculate detailed metrics for each report
            report_metrics = []
            for i, doc in enumerate(ordered_docs):
                total_steps = 0
                failed_steps = 0
                total_execution_time = 0
                
                for scenario in doc.get('scenarios', []):
                    for step in scenario.get('steps', []):
                        total_steps += 1
                        if step.get('result', {}).get('status') == 'failed':
                            failed_steps += 1
                        # Add step duration to total execution time
                        total_execution_time += step.get('result', {}).get('duration', 0)

                passed_steps = total_steps - failed_steps
                failure_rate = (failed_steps / total_steps * 100) if total_steps > 0 else 0
                pass_rate = (passed_steps / total_steps * 100) if total_steps > 0 else 0

                report_metrics.append({
                    "report_id": doc.get('_id'),
                    "total_steps": total_steps,
                    "failed_steps": failed_steps,
                    "passed_steps": passed_steps,
                    "failure_rate": round(failure_rate, 2),
                    "pass_rate": round(pass_rate, 2),
                    "total_execution_time_ms": total_execution_time,
                    "total_execution_time_formatted": f"{total_execution_time/1000:.2f} seconds"
                })

            # Update comparison data to use report IDs
            comparison = ProjectComparator.compare_projects(ordered_docs)
            
            # Ensure comparison is a dictionary
            if not isinstance(comparison, dict):
                comparison = {
                    'failure_rates': {},
                    'common_failures': [],
                    'scenario_comparison': {},
                    'tabular_comparison': []
                }
            
            # Generate tabular comparison
            all_steps = set()
            for doc in ordered_docs:
                for scenario in doc.get('scenarios', []):
                    for step in scenario.get('steps', []):
                        all_steps.add(step.get('name', 'Unnamed Step'))
            
            if all_steps:
                table = [
                    f"| Step | {report_ids[0]} Status | {report_ids[1]} Status | Observation |",
                    "|------|----------------------|----------------------|-------------|"
                ]
                
                for step in sorted(all_steps):
                    row = [f"| {step} "]
                    statuses = {}
                    observations = []
                    
                    for i, doc in enumerate(ordered_docs):
                        found = False
                        for scenario in doc.get('scenarios', []):
                            for s in scenario.get('steps', []):
                                if s.get('name') == step:
                                    status = "✅ Passed" if s.get('result', {}).get('status') == 'passed' else "❌ Failed"
                                    statuses[f"report{i+1}"] = status
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            statuses[f"report{i+1}"] = "–"
                    
                    row.append(f"| {statuses.get('report1', '–')} ")
                    row.append(f"| {statuses.get('report2', '–')} ")
                    
                    # Generate observations
                    if statuses.get('report1') == statuses.get('report2'):
                        if statuses.get('report1') == "✅ Passed":
                            observations.append("Stable across both")
                        elif statuses.get('report1') == "❌ Failed":
                            observations.append("Consistent failure")
                    else:
                        if "browser" in step.lower():
                            observations.append("Browser initialization issue")
                        elif "navigate" in step.lower():
                            observations.append("Page load problem")
                        elif "click" in step.lower():
                            observations.append("Element interaction issue")
                        elif "enter" in step.lower():
                            observations.append("Input field problem")
                        elif "wait" in step.lower():
                            observations.append("Timing issue")
                        else:
                            observations.append("Behavior difference")
                    
                    row.append(f"| {' '.join(observations)} |")
                    table.append(''.join(row))
                
                tabular_comparison = "\n".join(table)
            else:
                tabular_comparison = f"""| Step | {report_ids[0]} Status | {report_ids[1]} Status | Observation |
|------|----------------------|----------------------|-------------|
| No steps found for comparison | - | - | - |"""

            # Calculate common failures
            failure_counts = {}
            failure_details = []
            for doc in ordered_docs:
                for scenario in doc.get('scenarios', []):
                    for step in scenario.get('steps', []):
                        if step.get('result', {}).get('status') == 'failed':
                            step_name = step.get('name', 'Unnamed Step')
                            failure_counts[step_name] = failure_counts.get(step_name, 0) + 1
                            # Collect detailed failure information for AI analysis
                            failure_details.append({
                                'report_id': doc.get('_id'),
                                'scenario': scenario.get('name', 'Unnamed Scenario'),
                                'step': step_name,
                                'error': step.get('result', {}).get('error', 'No error message'),
                                'timestamp': step.get('result', {}).get('timestamp', ''),
                                'duration': step.get('result', {}).get('duration', 0)
                            })
            
            common_failures = [
                {'step_name': step, 'failure_count': count}
                for step, count in failure_counts.items()
                if count > 1
            ]
            common_failures.sort(key=lambda x: x['failure_count'], reverse=True)

            # Generate rectification steps and best practices
            try:
                # Generate rectification steps
                rectification_prompt = f"""As a test automation expert, analyze these test failures and provide detailed rectification steps:

Failure Details:
{json.dumps(failure_details, indent=2)}

For each failure, provide:
1. Root Cause Analysis
   - Technical explanation of why the failure occurs
   - Environmental factors that might contribute
   - Dependencies that could be affected

2. Immediate Fix Steps
   - Step-by-step instructions to fix the issue
   - Code changes or configuration updates needed
   - Testing steps to verify the fix

3. Verification Process
   - How to verify the fix is working
   - What metrics to check
   - How to ensure no regression

Format the response in clear sections with bullet points for easy reading."""

                print("Generating rectification steps...")
                rectification_response = model.generate_content(rectification_prompt)
                if rectification_response and hasattr(rectification_response, 'text'):
                    rectification_steps = rectification_response.text.strip()
                    if not rectification_steps:
                        raise ValueError("Empty response from AI model")
                else:
                    raise ValueError("Invalid response from AI model")
                print("Rectification steps generated successfully")

                # Generate best practices
                best_practices_prompt = f"""Based on these test failures, provide detailed best practices and prevention measures:

Failure Details:
{json.dumps(failure_details, indent=2)}

For each type of failure found, provide:

1. Specific Rectification Steps
   - Exact steps to fix each type of failure
   - Code changes or configuration updates needed
   - How to verify the fix works

2. Prevention Measures
   - How to prevent these specific failures in future tests
   - Changes needed in test design
   - Environment setup improvements

3. Monitoring Recommendations
   - What specific metrics to track for these failures
   - How to detect similar issues early
   - Alert thresholds to set

4. Team Guidelines
   - Specific code review checkpoints for these issues
   - Testing procedures to catch these failures
   - Documentation requirements for these scenarios

Format each failure type as a separate section with clear headings and numbered steps."""

                print("Generating best practices...")
                best_practices_response = model.generate_content(best_practices_prompt)
                if best_practices_response and hasattr(best_practices_response, 'text'):
                    best_practices = best_practices_response.text.strip()
                    if not best_practices:
                        raise ValueError("Empty response from AI model")
                else:
                    raise ValueError("Invalid response from AI model")
                print("Best practices generated successfully")

            except Exception as e:
                print(f"Error generating AI analysis: {str(e)}")
                # Generate basic fallback content if AI fails
                rectification_steps = "Failed to generate specific rectification steps. Please check the test logs and verify element locators."
                best_practices = """Best Practices for Test Automation:

1. Test Design
   - Use clear, descriptive test names
   - Implement proper test isolation
   - Follow the Arrange-Act-Assert pattern
   - Keep tests independent and atomic

2. Error Handling
   - Implement robust error handling
   - Add proper logging and reporting
   - Use appropriate wait strategies
   - Handle dynamic elements properly

3. Maintenance
   - Regular test suite maintenance
   - Update test data regularly
   - Monitor test execution times
   - Review and update selectors"""

            return JSONResponse(
                content={
                    "status": "success",
                    "report_ids": report_ids,
                    "summary": {
                        "total_steps_diff": abs(report_metrics[0]['total_steps'] - report_metrics[1]['total_steps']),
                        "failure_rate_diff": abs(report_metrics[0]['failure_rate'] - report_metrics[1]['failure_rate']),
                        "pass_rate_diff": abs(report_metrics[0]['pass_rate'] - report_metrics[1]['pass_rate']),
                        "execution_time_diff": abs(report_metrics[0]['total_execution_time_ms'] - report_metrics[1]['total_execution_time_ms']),
                        "common_failures_count": len(common_failures)
                    },
                    "detailed_metrics": {
                        "report_metrics": report_metrics
                    },
                    "comparison_data": {
                        "common_failures": common_failures
                    },
                    "tabular_comparison": tabular_comparison,
                    "rectification_steps": rectification_steps,
                    "best_practices": best_practices,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                status_code=200
            )

        except PyMongoError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
        finally:
            if client:
                client.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )

@router.get("/compare-projects")
async def compare_projects(
    project_ids: List[str] = Query(..., description="Exactly two project IDs to compare"),
    days_back: int = 7
):
    """Main endpoint for comparing two projects"""
    # ... rest of the function implementation ...

@router.get("/compare-projects/sorted-text", tags=["Comparison"], response_class=PlainTextResponse)
async def compare_projects_sorted_text(
    project_ids: List[str] = Query(..., description="Exactly two project IDs to compare"),
    days_back: int = Query(7, ge=1, description="Number of days to look back")
):
    """Get a sorted plain text comparison summary between test projects."""
    # ... rest of the function implementation ...

@router.get("/compare-projects/ai", tags=["Comparison"], response_class=PlainTextResponse)
async def compare_projects_ai_summary(
    project_ids: List[str] = Query(..., description="Exactly two project IDs to compare"),
    days_back: int = Query(7, ge=1, description="Number of days to look back")
):
    """Get an AI-generated 2-3 line human-friendly comparison summary between test projects."""
    try:
        client = None
        try:
            client = MongoDBManager.get_client()
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]

            # Build query to handle both ObjectId and UUID-style IDs
            query = {
                "_class": "pulse_qa_api.entity.GenerateReport",
                "$or": [
                    {"_id": ObjectId(pid) if ObjectId.is_valid(pid) else None} for pid in project_ids
                ] + [
                    {"projectId": pid} for pid in project_ids
                ]
            }
            
            # Clean up query by removing None conditions
            query["$or"] = [cond for cond in query["$or"] if list(cond.values())[0] is not None]

            if days_back:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
                query["scenarios.start_timestamp"] = {"$gte": cutoff_date.isoformat()}

            # Find all matching projects
            docs = list(collection.find(query))
            print(f"Found {len(docs)} matching documents")

            if len(docs) < 2:
                raise HTTPException(
                    status_code=404,
                    detail=f"Could not find enough projects. Found {len(docs)}/2"
                )

            # If we found more than 2 projects, select the most relevant ones
            if len(docs) > 2:
                print("Found more than 2 projects, selecting the most relevant ones...")
                # Sort by timestamp to get the most recent projects
                docs.sort(key=lambda x: x.get('scenarios', [{}])[0].get('start_timestamp', ''), reverse=True)
                # Take the first two projects
                docs = docs[:2]
                print(f"Selected 2 most recent projects for comparison")

            # Convert ObjectId to string for response
            for doc in docs:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                if 'projectId' in doc and isinstance(doc['projectId'], ObjectId):
                    doc['projectId'] = str(doc['projectId'])

            # Calculate failure rates and prepare data for Gemini
            project_stats = []
            for project in docs:
                project_name = project.get('projectName', 'Unnamed Project')
                total_steps = 0
                failed_steps = 0
                common_failures = {}
                
                if 'scenarios' in project:
                    for scenario in project['scenarios']:
                        if 'steps' in scenario:
                            for step in scenario['steps']:
                                total_steps += 1
                                if step.get('result', {}).get('status') == 'failed':
                                    failed_steps += 1
                                    step_name = step.get('name', 'unnamed_step')
                                    common_failures[step_name] = common_failures.get(step_name, 0) + 1
                
                failure_rate = (failed_steps / total_steps * 100) if total_steps > 0 else 0
                
                # Get top 3 common failures
                top_failures = []
                if common_failures:
                    top_failures = sorted(common_failures.items(), key=lambda x: x[1], reverse=True)[:3]
                
                project_stats.append({
                    "name": project_name,
                    "id": str(project.get('_id')),
                    "total_steps": total_steps,
                    "failed_steps": failed_steps,
                    "failure_rate": round(failure_rate, 2),
                    "top_failures": [{"step": name, "count": count} for name, count in top_failures]
                })

            # Generate AI summary
            try:
                prompt = f"""You are an expert test analysis assistant. Create a concise 2-3 line summary comparing these two test projects. Focus on the most important differences and insights. Be direct and use simple language.

Project 1: {project_stats[0]['name']}
- Failure rate: {project_stats[0]['failure_rate']}% ({project_stats[0]['failed_steps']}/{project_stats[0]['total_steps']} steps failed)
- Top failures: {', '.join([f"{f['step']} ({f['count']} times)" for f in project_stats[0]['top_failures'][:2]])}

Project 2: {project_stats[1]['name']}
- Failure rate: {project_stats[1]['failure_rate']}% ({project_stats[1]['failed_steps']}/{project_stats[1]['total_steps']} steps failed)
- Top failures: {', '.join([f"{f['step']} ({f['count']} times)" for f in project_stats[1]['top_failures'][:2]])}

The summary should be exactly 2-3 short lines, focusing on the most important insights that would help developers quickly understand the difference between these test projects."""

                response = model.generate_content(prompt)
                ai_summary = response.text.strip()
                
                # Ensure we have 2-3 lines max
                lines = ai_summary.split('\n')
                if len(lines) > 3:
                    ai_summary = '\n'.join(lines[:3])
                
                return ai_summary
                
            except Exception as e:
                print(f"Error generating AI summary: {str(e)}")
                # Fallback if AI generation fails
                p1, p2 = project_stats[0], project_stats[1]
                
                # Sort projects by failure rate
                if p1['failure_rate'] < p2['failure_rate']:
                    p1, p2 = p2, p1
                    
                # Calculate difference
                diff = abs(p1['failure_rate'] - p2['failure_rate'])
                
                line1 = f"{p1['name']} has a {p1['failure_rate']}% failure rate vs {p2['name']}'s {p2['failure_rate']}% rate."
                line2 = f"Difference of {round(diff, 1)}% between projects with {p1['name']} showing {p1['failed_steps']} failures."
                
                if p1['top_failures'] and p2['top_failures']:
                    common_issues = set([f['step'] for f in p1['top_failures']]) & set([f['step'] for f in p2['top_failures']])
                    if common_issues:
                        line3 = f"Both projects fail on: {next(iter(common_issues))}."
                        return f"{line1}\n{line2}\n{line3}"
                
                return f"{line1}\n{line2}"

        except PyMongoError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
        finally:
            if client:
                client.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating AI summary: {str(e)}"
        )

@router.get("/project/failures-summary", tags=["Analysis"], response_class=PlainTextResponse)
async def project_failures_ai_summary(
    project_id: Optional[str] = Query(None, description="Project ID to analyze failures"),
    report_id: Optional[str] = Query(None, description="Report ID to analyze failures"),
    days_back: int = Query(7, ge=1, description="Number of days to look back")
):
    """Get an AI-generated summary focusing on project/report failures.
    
    Parameters:
    - project_id: ID of the project to analyze (optional)
    - report_id: ID of the specific report to analyze (optional)
    - days_back: Number of days to look back (default: 7)
    
    Note: If both project_id and report_id are provided, the endpoint will analyze both
    and provide a comparison of failures between the project and the specific report.
    """
    try:
        if not project_id and not report_id:
            raise HTTPException(
                status_code=400,
                detail="At least one of project_id or report_id must be provided"
            )

        client = None
        try:
            client = MongoDBManager.get_client()
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]
            
            # Initialize data structures for both project and report
            project_data = None
            report_data = None
            failure_details = []
            
            # Function to process a document and extract failure details
            def process_document(doc, source_type):
                doc_failures = []
                total_steps = 0
                failed_steps = 0
                
                for scenario in doc.get('scenarios', []):
                    for step in scenario.get('steps', []):
                        total_steps += 1
                        if step.get('result', {}).get('status') == 'failed':
                            failed_steps += 1
                            # Get the actual error message
                            error_message = step.get('result', {}).get('error', '')
                            if not error_message and step.get('result', {}).get('errorMessage'):
                                error_message = step.get('result', {}).get('errorMessage')
                            if not error_message and step.get('result', {}).get('message'):
                                error_message = step.get('result', {}).get('message')
                            
                            doc_failures.append({
                                'source_type': source_type,
                                'source_id': str(doc.get('_id')),
                                'source_name': doc.get('projectName', 'Unnamed Project'),
                                'scenario': scenario.get('name', 'Unnamed Scenario'),
                                'step': step.get('name', 'Unnamed Step'),
                                'error': error_message or 'No error message',
                                'timestamp': step.get('result', {}).get('timestamp', '')
                            })
                
                failure_rate = (failed_steps / total_steps * 100) if total_steps > 0 else 0
                return {
                    'total_steps': total_steps,
                    'failed_steps': failed_steps,
                    'failure_rate': round(failure_rate, 2),
                    'failures': doc_failures
                }
            
            # Process project if provided
            if project_id:
                project_query = {
                    "_class": "pulseQA_Web.entity.GenerateReport",
                    "$or": [
                        {"_id": ObjectId(project_id) if ObjectId.is_valid(project_id) else None},
                        {"projectId": project_id}
                    ]
                }
                project_query["$or"] = [cond for cond in project_query["$or"] if list(cond.values())[0] is not None]
                
                if days_back:
                    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
                    project_query["scenarios.start_timestamp"] = {"$gte": cutoff_date.isoformat()}
                
                project_doc = collection.find_one(project_query)
                if project_doc:
                    project_data = process_document(project_doc, "project")
                    failure_details.extend(project_data['failures'])
            
            # Process report if provided
            if report_id:
                report_query = {
                    "_class": "pulseQA_Web.entity.GenerateReport",
                    "$or": [
                        {"_id": ObjectId(report_id) if ObjectId.is_valid(report_id) else None},
                        {"reportId": report_id}
                    ]
                }
                report_query["$or"] = [cond for cond in report_query["$or"] if list(cond.values())[0] is not None]
                
                if days_back:
                    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
                    report_query["scenarios.start_timestamp"] = {"$gte": cutoff_date.isoformat()}
                
                report_doc = collection.find_one(report_query)
                if report_doc:
                    report_data = process_document(report_doc, "report")
                    failure_details.extend(report_data['failures'])
            
            if not project_data and not report_data:
                raise HTTPException(
                    status_code=404,
                    detail="Could not find any data for the provided IDs"
                )

            # Generate AI analysis
            try:
                if project_data and report_data:
                    # Compare project and report
                    prompt = f"""Analyze and compare these test failures between a project and a specific report:

Project Data:
- Total Steps: {project_data['total_steps']}
- Failed Steps: {project_data['failed_steps']}
- Failure Rate: {project_data['failure_rate']}%

Report Data:
- Total Steps: {report_data['total_steps']}
- Failed Steps: {report_data['failed_steps']}
- Failure Rate: {report_data['failure_rate']}%

Failure Details:
{json.dumps(failure_details, indent=2)}

Provide a concise 2-3 line summary comparing the failures between the project and report, focusing on:
1. Key differences in failure patterns
2. Most critical issues
3. Notable improvements or regressions"""

                else:
                    # Single source analysis
                    data = project_data or report_data
                    source_type = "Project" if project_data else "Report"
                    prompt = f"""Analyze these test failures and provide a concise 2-3 line summary:

{source_type} Data:
- Total Steps: {data['total_steps']}
- Failed Steps: {data['failed_steps']}
- Failure Rate: {data['failure_rate']}%

Failure Details:
{json.dumps(failure_details, indent=2)}

Provide a concise 2-3 line summary focusing on:
1. Most critical issues
2. Failure patterns
3. Impact on test stability"""

                response = model.generate_content(prompt)
                failure_summary = response.text.strip()

                # If the summary is empty or failed, provide a default summary
                if not failure_summary:
                    if project_data and report_data:
                        failure_summary = f"Project: {project_data['failed_steps']}/{project_data['total_steps']} steps failed ({project_data['failure_rate']}%). Report: {report_data['failed_steps']}/{report_data['total_steps']} steps failed ({report_data['failure_rate']}%)."
                    else:
                        data = project_data or report_data
                        failure_summary = f"Found {data['failed_steps']} failures out of {data['total_steps']} steps ({data['failure_rate']}% failure rate)."

            except Exception as e:
                print(f"Error generating AI analysis: {str(e)}")
                if project_data and report_data:
                    failure_summary = f"Project: {project_data['failed_steps']}/{project_data['total_steps']} steps failed ({project_data['failure_rate']}%). Report: {report_data['failed_steps']}/{report_data['total_steps']} steps failed ({report_data['failure_rate']}%). AI analysis failed: {str(e)}"
                else:
                    data = project_data or report_data
                    failure_summary = f"Found {data['failed_steps']} failures out of {data['total_steps']} steps ({data['failure_rate']}% failure rate). AI analysis failed: {str(e)}"

            return PlainTextResponse(
                content=failure_summary,
                status_code=200
            )

        except PyMongoError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
        finally:
            if client:
                client.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        client = None
        try:
            client = MongoDBManager.get_client()
            client.admin.command('ping')
            model.generate_content("Health check")
            return {
                "status": "healthy",
                "services": ["mongodb", "gemini"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Service unavailable: {str(e)}"
            )
        finally:
            if client:
                client.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

@router.post("/report-chat", tags=["Chat"])
async def report_chat(
    report_ids: List[str] = Query(..., description="List of report IDs to analyze"),
    question: str = Query(..., description="User's question about the reports"),
    days_back: int = Query(7, ge=1, description="Number of days to look back")
):
    """Chat endpoint for asking questions about specific reports."""
    try:
        client = None
        try:
            client = MongoDBManager.get_client()
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]

            # Build query to handle both ObjectId and UUID-style IDs
            query = {
                "_class": "pulseQA_Web.entity.GenerateReport",
                "$or": [
                    {"_id": ObjectId(rid) if ObjectId.is_valid(rid) else None} for rid in report_ids
                ] + [
                    {"reportId": rid} for rid in report_ids
                ]
            }
            
            # Clean up query by removing None conditions
            query["$or"] = [cond for cond in query["$or"] if list(cond.values())[0] is not None]

            if days_back:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
                query["scenarios.start_timestamp"] = {"$gte": cutoff_date.isoformat()}

            # Find all matching reports
            docs = list(collection.find(query))
            
            if not docs:
                raise HTTPException(
                    status_code=404,
                    detail="No reports found with the provided IDs"
                )

            # Convert ObjectId to string for response
            for doc in docs:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                if 'reportId' in doc and isinstance(doc['reportId'], ObjectId):
                    doc['reportId'] = str(doc['reportId'])

            # Prepare report data for AI analysis
            report_data = []
            for doc in docs:
                report_name = doc.get('reportName', 'Unnamed Report')
                total_steps = 0
                failed_steps = 0
                scenarios = []
                
                for scenario in doc.get('scenarios', []):
                    scenario_data = {
                        'name': scenario.get('name', 'Unnamed Scenario'),
                        'steps': [],
                        'status': 'passed'
                    }
                    
                    for step in scenario.get('steps', []):
                        total_steps += 1
                        step_data = {
                            'name': step.get('name', 'Unnamed Step'),
                            'status': step.get('result', {}).get('status', 'unknown'),
                            'error': step.get('result', {}).get('error', ''),
                            'duration': step.get('result', {}).get('duration', 0)
                        }
                        
                        if step_data['status'] == 'failed':
                            failed_steps += 1
                            scenario_data['status'] = 'failed'
                        
                        scenario_data['steps'].append(step_data)
                    
                    scenarios.append(scenario_data)
                
                failure_rate = (failed_steps / total_steps * 100) if total_steps > 0 else 0
                
                report_data.append({
                    'name': report_name,
                    'total_steps': total_steps,
                    'failed_steps': failed_steps,
                    'failure_rate': round(failure_rate, 2),
                    'scenarios': scenarios
                })

            # Generate AI response
            try:
                prompt = f"""You are an expert test analysis assistant. Answer the following question about these test reports.
Only use the information provided in the reports. If the question cannot be answered with the available data, say so.

Question: {question}

Report Data:
{json.dumps(report_data, indent=2)}

Provide a clear, concise answer based only on the report data. If the question is about something not covered in the reports, explain that the information is not available in the provided reports."""

                response = model.generate_content(prompt)
                ai_response = response.text.strip()

                if not ai_response:
                    raise ValueError("Empty response from AI model")

                return JSONResponse(
                    content={
                        "status": "success",
                        "question": question,
                        "answer": ai_response,
                        "report_data": report_data,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    status_code=200
                )

            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error generating AI response: {str(e)}"
                )

        except PyMongoError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
        finally:
            if client:
                client.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat analysis failed: {str(e)}"
        )

@router.get("/get-rectification-steps", tags=["Analysis"])
async def get_rectification_steps(
    report_id: str = Query(..., description="Report ID to analyze failures"),
    days_back: int = Query(7, ge=1, description="Number of days to look back")
):
    """Get concise rectification steps for failed test cases in a specific report."""
    try:
        client = None
        try:
            client = MongoDBManager.get_client()
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]

            # Build query to handle both ObjectId and UUID-style IDs
            query = {
                "_class": "pulseQA_Web.entity.GenerateReport",
                "$or": [
                    {"_id": ObjectId(report_id) if ObjectId.is_valid(report_id) else None},
                    {"reportId": report_id}
                ]
            }
            
            # Clean up query by removing None conditions
            query["$or"] = [cond for cond in query["$or"] if list(cond.values())[0] is not None]

            if days_back:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
                query["scenarios.start_timestamp"] = {"$gte": cutoff_date.isoformat()}

            # Find the report
            doc = collection.find_one(query)
            
            if not doc:
                raise HTTPException(
                    status_code=404,
                    detail=f"Could not find report with ID: {report_id}"
                )

            # Convert ObjectId to string for response
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            if 'reportId' in doc and isinstance(doc['reportId'], ObjectId):
                doc['reportId'] = str(doc['reportId'])

            # Extract simplified failure details
            failure_details = []
            total_steps = 0
            failed_steps = 0
            
            for scenario in doc.get('scenarios', []):
                for step in scenario.get('steps', []):
                    total_steps += 1
                    if step.get('result', {}).get('status') == 'failed':
                        failed_steps += 1
                        # Get the actual error message from the result
                        error_message = step.get('result', {}).get('error', '')
                        if not error_message and step.get('result', {}).get('errorMessage'):
                            error_message = step.get('result', {}).get('errorMessage')
                        if not error_message and step.get('result', {}).get('message'):
                            error_message = step.get('result', {}).get('message')
                        
                        failure_details.append({
                            'step': step.get('name', 'Unnamed Step'),
                            'error': error_message or 'Element not found or not interactable'
                        })

            # Calculate failure rate
            failure_rate = (failed_steps / total_steps * 100) if total_steps > 0 else 0

            # Generate concise AI analysis for rectification steps
            try:
                prompt = f"""As a test automation expert, provide a brief two-line rectification step for each test failure.
For each failure, provide exactly two lines:
1. First line: What is the issue (based on the error message)
2. Second line: How to fix it

Failure Details:
{json.dumps(failure_details, indent=2)}

Format each failure's rectification as:
Failure: [step name]
Issue: [one line describing the specific issue based on the error]
Fix: [one line with the specific fix for this error]"""

                print("Generating rectification steps...")
                response = model.generate_content(prompt)
                if response and hasattr(response, 'text'):
                    rectification_steps = response.text.strip()
                    if not rectification_steps:
                        raise ValueError("Empty response from AI model")
                else:
                    raise ValueError("Invalid response from AI model")
                print("Rectification steps generated successfully")

            except Exception as e:
                print(f"Error generating rectification steps: {str(e)}")
                # Generate basic rectification steps if AI fails
                rectification_steps = "Failed to generate specific rectification steps. Please check the test logs and verify element locators."

            return JSONResponse(
                content={
                    "status": "success",
                    "report_id": doc.get('_id'),
                    "failure_summary": {
                        "total_steps": total_steps,
                        "failed_steps": failed_steps,
                        "failure_rate": round(failure_rate, 2)
                    },
                    "failures": failure_details,
                    "rectification_steps": rectification_steps
                },
                status_code=200
            )

        except PyMongoError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
        finally:
            if client:
                client.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

# New models for classification
class CacheStatsResponse(BaseModel):
    total_entries: int
    smoke_count: int
    regression_only_count: int

class FileResult(BaseModel):
    filename: str
    content: str

class ClassificationResponse(BaseModel):
    files: List[FileResult]

class TestClassificationCache:
    """Simple file-based cache for test classifications"""
    
    def __init__(self, cache_file="test_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, str]:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            pass
    
    def get_hash(self, content: str) -> str:
        """Generate hash for scenario content"""
        normalized = re.sub(r'\s+', ' ', content.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, content: str) -> str:
        """Get cached classification"""
        hash_key = self.get_hash(content)
        return self.cache.get(hash_key)
    
    def set(self, content: str, classification: str):
        """Cache classification"""
        hash_key = self.get_hash(content)
        self.cache[hash_key] = classification
        self._save_cache()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache),
            'smoke_count': sum(1 for v in self.cache.values() if 'smoke_regression' in v.lower()),
            'regression_only_count': sum(1 for v in self.cache.values() if v == 'regression')
        }

class DualTagTestClassifier:
    """Enhanced Test classifier for dual tagging scenarios"""
    
    def __init__(self):
        self.cache = TestClassificationCache()
        self.batch_prompt = """
Analyze the following Gherkin test scenarios and classify each one:

CLASSIFICATION RULES:
- ALL scenarios MUST have @regression tag (mandatory for all tests)
- CRITICAL scenarios get BOTH @smoke AND @regression tags (dual tagging)
- Add @smoke tag for scenarios that test essential functionality:
  * Browser operations: open browser, close browser, maximize window, minimize window
  * Navigation: navigate to URL, go to page, visit URL, load page
  * Core authentication: login, logout, sign in, authentication
  * Application startup: launch app, start application, initialize
  * Basic functionality: load content, verify elements, check page loads
  * System health: connectivity checks, basic availability tests
  * Essential user flows: home page access, main menu navigation

OUTPUT FORMAT:
- Regular scenarios: "regression"
- Critical scenarios: "smoke_regression"

SCENARIOS TO ANALYZE:
{scenarios}

PROVIDE CLASSIFICATIONS (one per line):
"""

    def classify_scenarios_batch(self, scenarios: List[str], batch_size: int = 8) -> List[str]:
        """Classify scenarios in batches with improved logic"""
        classifications = []
        uncached_scenarios = []
        uncached_indices = []
        
        for i, scenario in enumerate(scenarios):
            cached_result = self.cache.get(scenario)
            if cached_result:
                classifications.append(cached_result)
            else:
                classifications.append(None)
                uncached_scenarios.append(scenario)
                uncached_indices.append(i)
        
        if not uncached_scenarios:
            return classifications
        
        for batch_idx in range(0, len(uncached_scenarios), batch_size):
            batch = uncached_scenarios[batch_idx:batch_idx + batch_size]
            batch_indices = uncached_indices[batch_idx:batch_idx + batch_size]
            
            scenario_list = ""
            for j, scenario in enumerate(batch, 1):
                lines = scenario.split('\n')
                title_line = lines[0] if lines else scenario[:100]
                key_steps = []
                for line in lines[1:]:
                    stripped = line.strip()
                    if stripped.startswith(('Given', 'When', 'Then', 'And', 'But')):
                        key_steps.append(stripped)
                        if len(key_steps) >= 4:
                            break
                
                compact_scenario = title_line
                if key_steps:
                    compact_scenario += '\n' + '\n'.join(key_steps)
                
                scenario_list += f"{j}. {compact_scenario}\n\n"
            
            try:
                prompt = self.batch_prompt.format(scenarios=scenario_list.strip())
                response = model.generate_content(prompt)
                response_text = response.text.strip()
                
                batch_classifications = self._parse_ai_response(response_text, len(batch))
                
                if len(batch_classifications) != len(batch):
                    batch_classifications = [self._rule_based_classify(scenario) for scenario in batch]
                
                for j, (scenario, classification) in enumerate(zip(batch, batch_classifications)):
                    idx = batch_indices[j]
                    classifications[idx] = classification
                    self.cache.set(scenario, classification)
                
            except Exception as e:
                print(f"AI classification failed: {e}")
                for j, scenario in enumerate(batch):
                    idx = batch_indices[j]
                    classification = self._rule_based_classify(scenario)
                    classifications[idx] = classification
                    self.cache.set(scenario, classification)
            
            time.sleep(0.2)
        
        return classifications

    def _parse_ai_response(self, response_text: str, expected_count: int) -> List[str]:
        """Parse AI response to extract classifications"""
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        classifications = []
        
        for line in lines:
            line_lower = line.lower()
            if ('smoke_regression' in line_lower or 
                ('smoke' in line_lower and 'regression' in line_lower) or
                line_lower == '@smoke' or
                '@smoke' in line_lower):
                classifications.append('smoke_regression')
            elif (line_lower == 'regression' or 
                  line_lower == '@regression' or
                  (line_lower.endswith('regression') and 'smoke' not in line_lower)):
                classifications.append('regression')
            elif len(classifications) < expected_count:
                if any(keyword in line_lower for keyword in ['browser', 'navigate', 'login', 'open', 'close']):
                    classifications.append('smoke_regression')
                else:
                    classifications.append('regression')
        
        return classifications[:expected_count]

    def _rule_based_classify(self, scenario: str) -> str:
        """Enhanced rule-based classification as fallback"""
        scenario_lower = scenario.lower()
        
        critical_keywords = [
            'open browser', 'close browser', 'browser', 'maximize window', 'minimize window',
            'navigate to', 'go to url', 'navigate to url', 'visit url', 'load url',
            'open url', 'access url'
        ]
        
        important_keywords = [
            'login', 'log in', 'sign in', 'authentication', 'authenticate',
            'startup', 'launch', 'initialize', 'start application',
            'home page', 'main page', 'landing page', 'main menu',
            'load dynamic content', 'verify element', 'page loads', 'content loads',
            'application loads', 'connectivity check', 'health check'
        ]
        
        for keyword in critical_keywords:
            if keyword in scenario_lower:
                return 'smoke_regression'
        
        for keyword in important_keywords:
            if keyword in scenario_lower:
                return 'smoke_regression'
        
        return 'regression'

def extract_scenarios(content: str) -> List[Dict[str, Any]]:
    """Extract individual scenarios from a feature file"""
    scenarios = []
    lines = content.split('\n')
    current_scenario = []
    in_scenario = False
    scenario_start_line = 0
    
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        
        if (stripped_line.startswith('Scenario:') or 
            stripped_line.startswith('Scenario Outline:') or
            re.match(r'^\s*@.*Scenario:', line) or
            re.match(r'^\s*Scenario\s+Outline:', stripped_line)):
            
            if in_scenario and current_scenario:
                scenario_content = '\n'.join(current_scenario)
                scenarios.append({
                    'content': scenario_content,
                    'start_line': scenario_start_line,
                    'end_line': i - 1,
                    'title': extract_scenario_title(current_scenario)
                })
            
            current_scenario = [line]
            in_scenario = True
            scenario_start_line = i
        elif in_scenario:
            current_scenario.append(line)
            
            if (stripped_line.startswith(('Feature:', 'Background:', 'Rule:')) and 
                not stripped_line.startswith(('Given', 'When', 'Then', 'And', 'But'))):
                current_scenario.pop()
                scenario_content = '\n'.join(current_scenario)
                scenarios.append({
                    'content': scenario_content,
                    'start_line': scenario_start_line,
                    'end_line': i - 1,
                    'title': extract_scenario_title(current_scenario)
                })
                in_scenario = False
                current_scenario = []
    
    if in_scenario and current_scenario:
        scenario_content = '\n'.join(current_scenario)
        scenarios.append({
            'content': scenario_content,
            'start_line': scenario_start_line,
            'end_line': len(lines) - 1,
            'title': extract_scenario_title(current_scenario)
        })
    
    return scenarios

def extract_scenario_title(scenario_lines: List[str]) -> str:
    """Extract clean scenario title"""
    for line in scenario_lines:
        stripped = line.strip()
        if stripped.startswith(('Scenario:', 'Scenario Outline:')):
            return stripped
    return scenario_lines[0].strip() if scenario_lines else "Unknown Scenario"

async def process_files_optimized(uploaded_files: List[UploadFile], classifier: DualTagTestClassifier, batch_size: int = 8):
    """Process multiple files with enhanced dual tag classification"""
    temp_dir = "temp_uploaded_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    all_file_results = []
    classified_files = []
    all_scenarios = []
    file_scenario_mapping = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.filename)
        
        with open(file_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except:
                with open(file_path, 'r', encoding='cp1252') as f:
                    content = f.read()
        
        scenarios = extract_scenarios(content)
        if scenarios:
            file_info = {
                'filename': uploaded_file.filename,
                'content': content,
                'scenarios': scenarios,
                'scenario_start_idx': len(all_scenarios)
            }
            file_scenario_mapping.append(file_info)
            
            for scenario in scenarios:
                all_scenarios.append(scenario['content'])
    
    if not all_scenarios:
        raise HTTPException(status_code=400, detail="No scenarios found in uploaded files")
    
    print(f"Classifying {len(all_scenarios)} scenarios...")
    classifications = classifier.classify_scenarios_batch(all_scenarios, batch_size)
    print(f"Classification complete: {sum(1 for c in classifications if 'smoke' in c.lower())} smoke scenarios")
    
    for file_info in file_scenario_mapping:
        start_idx = file_info['scenario_start_idx']
        file_scenarios = file_info['scenarios']
        
        file_classifications = classifications[start_idx:start_idx + len(file_scenarios)]
        
        normalized_classifications = []
        for c in file_classifications:
            if c in ['@smoke', 'smoke', 'smoke_regression']:
                normalized_classifications.append('smoke_regression')
            else:
                normalized_classifications.append('regression')
        
        annotated_content = generate_annotated_content(
            file_info['content'], 
            file_scenarios, 
            normalized_classifications
        )
        
        smoke_count = sum(1 for c in normalized_classifications if c == 'smoke_regression')
        regression_only_count = sum(1 for c in normalized_classifications if c == 'regression')
        
        file_results = {
            'filename': file_info['filename'],
            'content': annotated_content,
            'smoke_count': smoke_count,
            'regression_only_count': regression_only_count,
            'total_scenarios': len(file_scenarios)
        }
        all_file_results.append(file_results)
        
        output_path = os.path.join(temp_dir, f"classified_{file_info['filename']}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(annotated_content)
        classified_files.append(output_path)
    
    return all_file_results, classified_files

def generate_annotated_content(original_content: str, scenarios: List[Dict], classifications: List[str]) -> str:
    """Generate content with properly placed tags"""
    lines = original_content.split('\n')
    result_lines = []
    
    line_to_classification = {}
    for scenario, classification in zip(scenarios, classifications):
        line_to_classification[scenario['start_line']] = classification
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if i in line_to_classification:
            classification = line_to_classification[i]
            
            scenario_line = line.strip()
            if scenario_line.startswith(('Scenario:', 'Scenario Outline:')):
                indent = len(line) - len(line.lstrip())
                tag_indent = ' ' * indent
                
                if classification == 'smoke_regression':
                    result_lines.append(f"{tag_indent}@smoke")
                    result_lines.append(f"{tag_indent}@regression")
                else:
                    result_lines.append(f"{tag_indent}@regression")
        
        result_lines.append(line)
        i += 1
    
    return '\n'.join(result_lines)

# New endpoints for classification
@router.get("/classification/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get classification cache statistics"""
    classifier = DualTagTestClassifier()
    cache_stats = classifier.cache.get_stats()
    return cache_stats

@router.delete("/classification/cache")
async def clear_classification_cache():
    """Clear the classification cache"""
    try:
        os.remove("test_cache.json")
        return {"message": "Cache cleared successfully"}
    except FileNotFoundError:
        return {"message": "No cache to clear"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classification/classify", response_model=ClassificationResponse)
async def classify_files(
    files: List[UploadFile] = File(...),
    batch_size: int = 8
):
    """Classify uploaded feature files with dual tags"""
    temp_dir = "temp_uploaded_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        for file in files:
            if not file.filename.endswith(('.feature', '.gherkin')):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type: {file.filename}. Only .feature and .gherkin files are supported"
                )
        
        classifier = DualTagTestClassifier()
        all_file_results, classified_files = await process_files_optimized(files, classifier, batch_size)
        
        if not all_file_results:
            raise HTTPException(status_code=400, detail="No files were successfully processed")
        
        # Prepare response with only filename and content
        response_files = []
        for result in all_file_results:
            response_files.append(FileResult(
                filename=result['filename'],
                content=result['content']
            ))
        
        return ClassificationResponse(files=response_files)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass

@router.post("/classification/download")
async def download_classified_files(
    files: List[UploadFile] = File(...),
    batch_size: int = 8
):
    """Classify files and return as a zip download"""
    classifier = DualTagTestClassifier()
    classified_files = []
    
    try:
        all_file_results, classified_files = await process_files_optimized(
            files, classifier, batch_size
        )
        
        if not classified_files:
            raise HTTPException(status_code=400, detail="No files were successfully processed")
        
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in classified_files:
                zip_filename = os.path.basename(file_path)
                zipf.write(file_path, zip_filename)
        
        zip_buffer.seek(0)
        
        return FileResponse(
            zip_buffer,
            media_type="application/zip",
            filename=f"dual_tagged_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")
    
    finally:
        if classified_files:
            for file in classified_files:
                try:
                    os.remove(file)
                except:
                    pass
        try:
            os.rmdir("temp_uploaded_files")
        except:
            pass 
