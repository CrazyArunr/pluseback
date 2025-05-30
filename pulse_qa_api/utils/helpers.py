"""
Utility functions for the Pulse QA API
"""

import os
import json
from typing import Dict, List, Optional
from fastapi import HTTPException
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from pulse_qa_api.config import settings

def get_custom_client() -> OpenAI:
    """Get OpenAI client with custom configuration"""
    return OpenAI(
        base_url=settings.CUSTOM_API_BASE,
        api_key=settings.CUSTOM_API_KEY,
        default_headers=settings.CUSTOM_HEADERS
    )

def get_langchain_custom_llm() -> ChatOpenAI:
    """Get LangChain compatible custom LLM client"""
    return ChatOpenAI(
        openai_api_key=settings.CUSTOM_API_KEY,
        openai_api_base=settings.CUSTOM_API_BASE,
        model_name=settings.CUSTOM_MODEL,
        temperature=0.1,
        default_headers=settings.CUSTOM_HEADERS
    )

def get_embeddings() -> HuggingFaceEmbeddings:
    """Get HuggingFace embeddings"""
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

async def process_uploaded_file(content: str, expected_type: str) -> Dict:
    """Process uploaded file content and validate its structure"""
    try:
        data = json.loads(content)
        
        # Validation based on expected type
        if expected_type == "test_data":
            if not isinstance(data, dict) or "test_data" not in data:
                raise ValueError("Test data should be a dictionary with 'test_data' key")
            if not isinstance(data["test_data"], dict):
                raise ValueError("test_data value should be a dictionary")
                
        elif expected_type == "element_data":
            if not isinstance(data, dict) or "elements" not in data:
                raise ValueError("Element data should be a dictionary with 'elements' key")
            if not isinstance(data["elements"], list):
                raise ValueError("elements should be a list")
            for element in data["elements"]:
                if not isinstance(element, dict):
                    raise ValueError("Each element should be a dictionary")
                if "name" not in element or "type" not in element or "value" not in element:
                    raise ValueError("Each element should have name, type and value")
                
        elif expected_type == "app_data":
            if not isinstance(data, dict) or "application_config" not in data:
                raise ValueError("Application data should contain 'application_config'")
            if not isinstance(data["application_config"], dict):
                raise ValueError("application_config should be a dictionary")
            if "url" not in data["application_config"] or "browserType" not in data["application_config"]:
                raise ValueError("application_config should contain url and browserType")
                
        return data
        
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format")
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")

def extract_features_from_scenario(scenario_text: str) -> Dict:
    """Extract features from scenario text"""
    try:
        # Initialize the features structure
        features = {
            "test_data": {},
            "elements_data": [],
            "application_data": {
                "url": "",
                "browserType": "edge"
            }
        }
        
        # Extract URL
        import re
        url_match = re.search(r"navigate to the URL\s+(?:'([^']+)'|([^\n]+))", scenario_text)
        if url_match:
            features["application_data"]["url"] = url_match.group(1) or url_match.group(2)
        
        # Extract all elements and their values
        steps = scenario_text.split('\n')
        element_counter = 1
        
        for step in steps:
            step = step.strip()
            if not step:
                continue
                
            # Extract input values and locators
            if "enter" in step.lower():
                # Extract value
                value_match = re.search(r"enter\s+'([^']+)'", step)
                if value_match:
                    value = value_match.group(1)
                    # Determine field type based on value pattern
                    if "@" in value and "." in value:
                        features["test_data"]["email"] = value
                    elif value.startswith("Test@"):
                        features["test_data"]["password"] = value
                    elif value.endswith("@gmail.com"):
                        features["test_data"]["email"] = value
                    else:
                        features["test_data"][f"field{element_counter}"] = value
                
                # Extract locator
                locator_match = re.search(r'locator\s+(?:"([^"]+)"|\'([^\']+)\')', step)
                if locator_match:
                    locator = locator_match.group(1) or locator_match.group(2)
                    element_name = f"element{element_counter}"
                    
                    # Determine element type based on locator
                    if "type='email'" in locator.lower() or "type=\"email\"" in locator.lower():
                        element_name = "emailInput"
                    elif "type='password'" in locator.lower() or "type=\"password\"" in locator.lower():
                        element_name = "passwordInput"
                    elif "username" in locator.lower():
                        element_name = "usernameInput"
                    elif "next" in locator.lower():
                        element_name = "nextButton"
                    elif "profile" in locator.lower():
                        element_name = "profileElement"
                    
                    # Add element to elements_data if not already present
                    if not any(e["name"] == element_name for e in features["elements_data"]):
                        features["elements_data"].append({
                            "name": element_name,
                            "type": "XPATH",
                            "value": locator
                        })
                        element_counter += 1
        
        # Add default values if not found
        if not features["application_data"]["url"]:
            features["application_data"]["url"] = "https://demo.com/automation-practice-form"
        
        return features
    except Exception as e:
        raise ValueError(f"Error extracting features from scenario: {str(e)}") 
