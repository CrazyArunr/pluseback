"""
POS API for test scenario generation and analysis
"""

import re  # Add missing import
import google.generativeai as genai
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import json
import os
from datetime import datetime, timezone
from ..config import settings
from ..utils.helpers import (
    get_custom_client,
    get_langchain_custom_llm,
    get_embeddings,
    process_uploaded_file,
    extract_features_from_scenario
)

# Create router
router = APIRouter()

# Define CUSTOM_MODEL
CUSTOM_MODEL = "llama3_1"

# Define model
model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
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

# Pydantic models for request/response
class ChatRequest(BaseModel):
    query: str
    test_data: Optional[Dict] = None
    element_data: Optional[Dict] = None
    app_data: Optional[Dict] = None
    save_as_feature: Optional[bool] = True

class StepPredictionRequest(BaseModel):
    current_step: str
    previous_steps: Optional[List[str]] = None
    context: Optional[Dict] = None

class DataGenerationRequest(BaseModel):
    data_type: str
    description: str

class ScenarioRequest(BaseModel):
    prompt: str

class SimpleChatRequest(BaseModel):
    prompt: str  # Only the prompt is required

class FeatureFileRequest(BaseModel):
    feature_content: str

def generate_login_scenario(config: Dict, test_values: Dict) -> str:
    """Generate a login test scenario"""
    return f"""Given open browser
And navigate to the URL '{config['base_url']}'
And maximize the browser window
When enter '{test_values['username']}' into the element with locator '{config['elements']['usernameInput']}'
And enter '{test_values['password']}' into the element with locator '{config['elements']['passwordInput']}'
And click on the element with locator '{config['elements']['loginButton']}'
Then wait for the element with locator '{config['elements']['dashboardElement']}' to be visible
And the element with locator '{config['elements']['successMessage']}' should have text 'Login successful'
And close the browser"""

def generate_registration_scenario(config: Dict, test_values: Dict) -> str:
    """Generate a registration test scenario"""
    return f"""Given open browser
And navigate to the URL '{config['base_url']}'
And maximize the browser window
When enter '{test_values['email']}' into the element with locator '{config['elements']['emailInput']}'
And enter '{test_values['password']}' into the element with locator '{config['elements']['passwordInput']}'
And click on the element with locator '{config['elements']['submitButton']}'
Then wait for the element with locator '{config['elements']['successMessage']}' to be visible
And the element with locator '{config['elements']['successMessage']}' should have text 'Registration successful'
And close the browser"""

def generate_profile_scenario(config: Dict, test_values: Dict) -> str:
    """Generate a profile update test scenario"""
    return f"""Given open browser
And navigate to the URL '{config['base_url']}'
And maximize the browser window
When click on the element with locator '{config['elements']['profileElement']}'
And enter '{test_values['email']}' into the element with locator '{config['elements']['emailInput']}'
And enter '{test_values['phone']}' into the element with locator '{config['elements']['phoneInput']}'
And click on the element with locator '{config['elements']['submitButton']}'
Then wait for the element with locator '{config['elements']['successMessage']}' to be visible
And the element with locator '{config['elements']['successMessage']}' should have text 'Profile updated successfully'
And close the browser"""

def generate_password_reset_scenario(config: Dict, test_values: Dict) -> str:
    """Generate a password reset test scenario"""
    return f"""Given open browser
And navigate to the URL '{config['base_url']}'
And maximize the browser window
When click on the element with locator '{config['elements']['resetButton']}'
And enter '{test_values['email']}' into the element with locator '{config['elements']['emailInput']}'
And click on the element with locator '{config['elements']['submitButton']}'
Then wait for the element with locator '{config['elements']['successMessage']}' to be visible
And the element with locator '{config['elements']['successMessage']}' should have text 'Password reset email sent'
And close the browser"""

def generate_generic_scenario(query: str, config: Dict, test_values: Dict) -> str:
    """Generate a generic test scenario based on the query"""
    return f"""Given open browser
And navigate to the URL '{config['base_url']}'
And maximize the browser window
When enter '{test_values['username']}' into the element with locator '{config['elements']['usernameInput']}'
And enter '{test_values['password']}' into the element with locator '{config['elements']['passwordInput']}'
And click on the element with locator '{config['elements']['submitButton']}'
Then wait for the element with locator '{config['elements']['successMessage']}' to be visible
And the element with locator '{config['elements']['successMessage']}' should have text 'Operation successful'
And close the browser"""

def save_as_feature_file(scenario: str, scenario_type: str) -> str:
    """Save the generated scenario as a .feature file"""
    try:
        # Create features directory if it doesn't exist
        features_dir = os.path.join(os.getcwd(), "features")
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(features_dir, f"{scenario_type}_{timestamp}.feature")
        
        # Write to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(scenario)
        
        return filename
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving feature file: {str(e)}")
    
def generate_data_via_prompt(prompt, data_type):
    client = get_custom_client()
    
    if data_type == "test_data":
        system_msg = """You are a test data generator. Create comprehensive test data in JSON format with the following structure:
        {
            "test_data": {
                "field1": "sample_value1",
                "field2": "sample_value2",
                ...
            }
        }
        Include realistic test values for various scenarios."""
        example = {
            "test_data": {
                "username": "testuser",
                "password": "Test@123",
                "email": "testuser@example.com",
                "phone": "1234567890"
            }
        }
    elif data_type == "element_data":
        system_msg = """You are a web element data generator. Create comprehensive element data in JSON format with the following structure:
        {
            "elements": [
                {
                    "name": "element_name",
                    "type": "locator_type (XPATH/CSS/ID)",
                    "value": "locator_value"
                },
                ...
            ]
        }
        Include common web elements for a web application."""
        example = {
            "elements": [
                {
                    "name": "usernameInput",
                    "type": "XPATH",
                    "value": "//input[@id='username']"
                },
                {
                    "name": "loginButton",
                    "type": "CSS",
                    "value": "#login-button"
                }
            ]
        }
    else:  # app_data
        system_msg = """You are an application configuration generator. Create comprehensive app data in JSON format with the following structure:
        {
            "application_config": {
                "url": "application_url",
                "browserType": "browser_name",
                "other_config": "value"
            }
        }
        Include typical configuration for a web application."""
        example = {
            "application_config": {
                "url": "https://example.com/login",
                "browserType": "chrome",
                "timeout": 30,
                "headless": False
            }
        }
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Generate {data_type} based on this description: {prompt}. Here's an example of the format: {json.dumps(example)}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=CUSTOM_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.3
        )
        
        result = response.choices[0].message.content
        json_pattern = r'\{.*\}'
        json_match = re.search(json_pattern, result, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        return json.loads(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating {data_type}: {str(e)}")

def get_response(query: str, test_data: Optional[Dict] = None, element_data: Optional[Dict] = None, app_data: Optional[Dict] = None, save_as_feature: bool = True) -> Dict:
    # Greeting response
    if any(greeting in query.lower() for greeting in ["hi", "hello", "who created you", "who made you"]):
        return {"response": "Hello! I was created by the Pulse QA team to assist with testing scenarios and answering questions about the application."}
    
    # Check for testing scenarios
    if any(keyword in query.lower() for keyword in ["login", "register", "sign up", "profile", "logout", "password", "test", "scenario"]):
        # Check if files were uploaded with proper structure
        files_uploaded = all([
            test_data and isinstance(test_data, dict) and "test_data" in test_data,
            element_data and isinstance(element_data, dict) and "elements" in element_data,
            app_data and isinstance(app_data, dict) and "application_config" in app_data
        ])
        
        if files_uploaded:
            try:
                # Add default elements if not present
                default_elements = [
                    {"name": "loginButton", "type": "XPATH", "value": "//button[@type=\"submit\"]"},
                    {"name": "dashboardElement", "type": "XPATH", "value": "//div[@id=\"dashboard\"]"},
                    {"name": "titleElement", "type": "XPATH", "value": "//h1"},
                    {"name": "emailInput", "type": "XPATH", "value": "//input[@id=\"email\"]"},
                    {"name": "resetButton", "type": "XPATH", "value": "//button[@type=\"submit\"]"},
                    {"name": "successMessage", "type": "XPATH", "value": "//div[@class=\"success-message\"]"},
                    {"name": "errorMessage", "type": "XPATH", "value": "//div[@class=\"error-message\"]"},
                    {"name": "newPasswordInput", "type": "XPATH", "value": "//input[@id=\"new-password\"]"},
                    {"name": "confirmPasswordInput", "type": "XPATH", "value": "//input[@id=\"confirm-password\"]"},
                    {"name": "submitButton", "type": "XPATH", "value": "//button[@type=\"submit\"]"}
                ]
                
                # Merge default elements with provided elements
                existing_names = {e["name"] for e in element_data["elements"]}
                for element in default_elements:
                    if element["name"] not in existing_names:
                        element_data["elements"].append(element)
                
                # Extract configuration
                config = {
                    "browser": app_data["application_config"].get("browserType", "chrome"),
                    "base_url": app_data["application_config"].get("url", "https://example.com"),
                    "elements": {e["name"]: e["name"] for e in element_data["elements"]}
                }
                
                # Extract test values with defaults
                test_values = {
                    "username": test_data["test_data"].get("username", "testuser"),
                    "password": test_data["test_data"].get("password", "password123"),
                    "email": test_data["test_data"].get("email", "test@example.com"),
                    "phone": test_data["test_data"].get("phone", "1234567890")
                }
                
                # Route to appropriate scenario generator
                scenario_type = ""
                if "login" in query.lower():
                    scenario = generate_login_scenario(config, test_values)
                    scenario_type = "login"
                elif "register" in query.lower() or "registration" in query.lower():
                    scenario = generate_registration_scenario(config, test_values)
                    scenario_type = "registration"
                elif "profile" in query.lower():
                    scenario = generate_profile_scenario(config, test_values)
                    scenario_type = "profile"
                elif "password" in query.lower():
                    scenario = generate_password_reset_scenario(config, test_values)
                    scenario_type = "password_reset"
                else:
                    scenario = generate_generic_scenario(query, config, test_values)
                    scenario_type = "generic"
                
                # Format the response with Feature and Scenario structure
                formatted_scenario = f"""Feature: {scenario_type.title()} Functionality

Scenario: {scenario_type.title()} Test
{scenario}"""
                
                response = {"response": formatted_scenario}
                
                # Always save as feature file by default
                if scenario_type:
                    feature_file = save_as_feature_file(formatted_scenario, scenario_type)
                    response["feature_file"] = feature_file
                
                return response
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error generating scenario: {str(e)}")
        else:
            return {"response": generate_default_scenario(query)}
    else:
        return {"response": "I can help you with test scenarios. Please ask about login, registration, profile, or password reset scenarios."}

def extract_features_from_scenario(scenario_text):
    """Extract features from the scenario text"""
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
        
        # Extract URL - handle both quoted and unquoted URLs
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
                
                # Extract locator - handle both single and double quotes
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
            
            # Extract click elements
            elif "click" in step.lower():
                # Extract locator - handle both single and double quotes
                locator_match = re.search(r'locator\s+(?:"([^"]+)"|\'([^\']+)\')', step)
                if locator_match:
                    locator = locator_match.group(1) or locator_match.group(2)
                    element_name = f"button{element_counter}"
                    
                    # Determine button type based on locator
                    if "identifierNext" in locator.lower():
                        element_name = "identifierNextButton"
                    elif "passwordNext" in locator.lower():
                        element_name = "passwordNextButton"
                    elif "next" in locator.lower():
                        element_name = "nextButton"
                    elif "submit" in locator.lower():
                        element_name = "submitButton"
                    elif "login" in locator.lower():
                        element_name = "loginButton"
                    
                    # Add element to elements_data if not already present
                    if not any(e["name"] == element_name for e in features["elements_data"]):
                        features["elements_data"].append({
                            "name": element_name,
                            "type": "XPATH",
                            "value": locator
                        })
                        element_counter += 1
            
            # Extract wait elements
            elif "wait for" in step.lower():
                # Extract locator - handle both single and double quotes
                locator_match = re.search(r'locator\s+(?:"([^"]+)"|\'([^\']+)\')', step)
                if locator_match:
                    locator = locator_match.group(1) or locator_match.group(2)
                    element_name = f"waitElement{element_counter}"
                    
                    # Add element to elements_data if not already present
                    if not any(e["name"] == element_name for e in features["elements_data"]):
                        features["elements_data"].append({
                            "name": element_name,
                            "type": "XPATH",
                            "value": locator
                        })
                        element_counter += 1
            
            # Extract text verification elements
            elif "should have text" in step.lower():
                # Extract locator - handle both single and double quotes
                locator_match = re.search(r'locator\s+(?:"([^"]+)"|\'([^\']+)\')', step)
                text_match = re.search(r"text\s+'([^']+)'", step)
                if locator_match and text_match:
                    locator = locator_match.group(1) or locator_match.group(2)
                    text = text_match.group(1)
                    element_name = f"textElement{element_counter}"
                    
                    # Add element to elements_data if not already present
                    if not any(e["name"] == element_name for e in features["elements_data"]):
                        features["elements_data"].append({
                            "name": element_name,
                            "type": "XPATH",
                            "value": locator
                        })
                        element_counter += 1
                    
                    # Add text to test_data
                    features["test_data"][f"expectedText{element_counter}"] = text
        
        # Add default values if not found
        if not features["application_data"]["url"]:
            features["application_data"]["url"] = "https://demo.com/automation-practice-form"
        
        return features
    except Exception as e:
        raise ValueError(f"Error extracting features from scenario: {str(e)}")

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Get data directly from request without defaults
        test_data = request.test_data
        element_data = request.element_data
        app_data = request.app_data
        save_as_feature = request.save_as_feature

        # If app_data is not provided, try to generate it
        if not app_data:
            try:
                # Generate default application data
                app_data = {
                    "application_config": {
                        "url": "https://demo.com/automation-practice-form",
                        "browserType": "edge",
                        "timeout": 30,
                        "headless": False
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to generate application data: {str(e)}")

        # Validate data structure if data is provided
        if test_data and (not isinstance(test_data, dict) or "test_data" not in test_data):
            raise ValueError("Invalid test data structure")
        if element_data and (not isinstance(element_data, dict) or "elements" not in element_data):
            raise ValueError("Invalid element data structure")
        if app_data and (not isinstance(app_data, dict) or "application_config" not in app_data):
            raise ValueError("Invalid application data structure")

        # Get response using provided data
        try:
            response = get_response(
                request.query,
                test_data,
                element_data,
                app_data,
                save_as_feature
            )
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/generate-ai-data")
#@app.post("/generate-ai-data")
async def generate_ai_data_endpoint(request: DataGenerationRequest):
    """
    Endpoint for generating test, element, or application data using AI
    """
    try:
        # Validate data type
        if request.data_type not in ["test_data", "element_data", "app_data"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid data type. Must be one of: test_data, element_data, app_data"
            )
        
        # Generate data using AI
        generated_data = generate_data_via_prompt(request.description, request.data_type)
        
        if generated_data:
            return JSONResponse(content={
                "message": f"{request.data_type} generated successfully",
                "generated_data": generated_data
            })
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate {request.data_type}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def predict_next_step(current_step: str, previous_steps: List[str] = None, context: Dict = None) -> List[str]:
    """
    Predict the next likely BDD steps based on the current step.
    Uses a fixed template to ensure consistent predictions.
    Only requires the current_step parameter.
    """
    try:
        # Define the fixed template steps with default values
        template_steps = [
            "Given open browser",
            "And navigate to the URL",
            "And maximize the browser window",
            "When enter 'testuser' into the element with locator 'usernameInput'",
            "And enter 'Test@123' into the element with locator 'passwordInput'",
            "And click on the element with locator 'loginButton'",
            "Then wait for the element with locator 'dashboardElement' to be visible",
            "And the element with locator 'successMessage' should have text 'Login successful'",
            "And close the browser"
        ]
        
        # Find the current step's index in the template
        current_index = -1
        for i, step in enumerate(template_steps):
            # Compare only the action part of the step (ignore values)
            current_action = current_step.split("'")[0].strip().lower()
            template_action = step.split("'")[0].strip().lower()
            if current_action == template_action:
                current_index = i
                break
        
        if current_index == -1:
            # If current step not found in template, return next 3 steps from template
            return template_steps[:3]
        
        # Get the next 3 steps from the template
        next_steps = template_steps[current_index + 1:current_index + 4]
        
        # Ensure we always return exactly 3 predictions
        while len(next_steps) < 3:
            next_steps.append("")
        next_steps = next_steps[:3]
        
        return next_steps
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting next step: {str(e)}")

@router.post("/predict-next-step")
async def predict_next_step_endpoint(request: StepPredictionRequest):
    """
    Endpoint for predicting the next likely BDD steps based on the current step.
    Only requires the current_step parameter in the request.
    """
    try:
        predictions = predict_next_step(
            current_step=request.current_step,
            previous_steps=request.previous_steps,
            context=request.context
        )
        
        return {
            "predictions": predictions,
            "current_step": request.current_step
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-scenario", tags=["Scenario Generation"])
async def generate_scenario_endpoint(request: SimpleChatRequest):
    """
    Enhanced endpoint that handles both test scenario generation and general queries.
    Can respond to:
    1. Test scenario generation requests
    2. General knowledge questions
    3. Greetings and conversational queries
    Always emphasizes Pulse QA team and testing automation for creator-related queries.
    """
    try:
        prompt_lower = request.prompt.lower()
        
        # Handle greetings and creator-related queries with consistent responses
        greeting_keywords = {
            "hi": "Hello! I'm your test automation assistant created by the Pulse QA team. How can I help you today?",
            "hello": "Hi there! I'm an AI assistant created by Pulse QA team for test automation. I can help you with test scenarios or answer general questions. What would you like to know?",
            "hey": "Hey! I'm here to help with test automation, created by the Pulse QA team. What can I do for you?",
            "greetings": "Greetings! I'm your AI assistant created by Pulse QA team. I can help with test scenarios or answer questions. What would you like to know?",
            "who are you": "I'm an AI assistant created by the Pulse QA team. I specialize in test automation and can help with both test scenarios and general questions. I can explain concepts, provide information, and help create test scenarios.",
            "what can you do": "I can help you with:\n1. Creating test scenarios\n2. Answering questions about technology, software, and testing\n3. Explaining concepts and providing information\n4. Having general conversations\nJust ask me anything! I was created by Pulse QA team to assist with test automation.",
            "help": "I can help you with:\n1. Test scenario generation\n2. General knowledge questions\n3. Technology explanations\n4. Software testing concepts\n5. General conversation\nI was created by Pulse QA team to assist with test automation. Just ask me anything!",
            "who created you": "I was created by the Pulse QA team to assist with test automation and help users create test scenarios. I can also answer general questions while maintaining my focus on testing and automation.",
            "who made you": "I was made by the Pulse QA team specifically for test automation purposes. I'm here to help create test scenarios and assist with testing-related tasks.",
            "your creator": "I was created by the Pulse QA team to help with test automation and scenario generation. I'm designed to assist users in creating and managing test scenarios effectively.",
            "your purpose": "I was created by Pulse QA team with the primary purpose of assisting in test automation. I help create test scenarios, answer testing-related questions, and provide general assistance while maintaining focus on testing and automation.",
            "why were you created": "I was created by the Pulse QA team to help streamline the test automation process. My main purpose is to assist in creating test scenarios and providing testing-related support, though I can also help with general queries.",
            "what is your purpose": "My primary purpose, as created by the Pulse QA team, is to assist with test automation. I help create test scenarios, answer testing-related questions, and provide support for automation tasks. I can also help with general queries while maintaining my focus on testing."
        }

        # Check for greetings and creator-related queries
        for keyword, response in greeting_keywords.items():
            if keyword in prompt_lower:
                return {"response": response}

        # Check if this is a test scenario request
        test_keywords = ["test", "scenario", "feature", "gherkin", "bdd", "automation", "login", "register", 
                        "sign up", "profile", "password", "logout", "testing", "test case"]
        
        if any(keyword in prompt_lower for keyword in test_keywords):
            # Handle test scenario generation
            client = get_custom_client()
            
            prompt = f"""
            Generate a test scenario following this exact structure:
            
            Feature: <Feature Name>
            Scenario: <Scenario Title>
            Steps:
            1. Given open browser 'chrome'
            2. And navigate to the URL '<URL>'
            3. And maximize the browser window
            4. When enter '<value>' into the element with locator '<locator>'
            5. And enter '<value>' into the element with locator '<locator>'
            6. And click on the element with locator '<locator>'
            7. Then wait for the element with locator '<locator>' to be visible
            8. And the element with locator '<locator>' should have text '<text>'
            9. And close the browser

            Based on this request: {request.prompt}
            
            Follow these rules:
            1. Use exact step format as shown above
            2. Include proper locators (XPATH/CSS)
            3. Include realistic test data
            4. Follow the Given/When/Then structure
            5. Include proper element locators and values
            """
            
            response = client.chat.completions.create(
                model=CUSTOM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a test automation expert that creates detailed test scenarios following the exact structure provided."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.1
            )
            
            return {"response": response.choices[0].message.content}
        
        # Handle general queries using Gemini
        try:
            # Configure Gemini for general queries
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 1,
                    "top_k": 32,
                    "max_output_tokens": 1024,
                }
            )

            # Create a prompt for general queries that emphasizes Pulse QA team for creator-related questions
            general_prompt = f"""You are a helpful AI assistant created by the Pulse QA team for test automation. Provide a clear, informative, and conversational response to this question:
            {request.prompt}
            
            Guidelines:
            1. Be informative but conversational
            2. Use simple, clear language
            3. Provide relevant examples if helpful
            4. Keep the response concise but complete
            5. If the question is unclear, ask for clarification
            6. If you don't know something, admit it
            7. If the question is about who created you or your purpose, always mention that you were created by Pulse QA team for test automation
            8. Maintain focus on testing and automation in your responses
            """

            # Generate response for general query
            response = model.generate_content(general_prompt)
            response_text = response.text.strip()

            # Additional check for creator-related queries that might have been missed
            creator_keywords = ["who created", "who made", "your creator", "your purpose", "why were you created", "what is your purpose"]
            if any(keyword in prompt_lower for keyword in creator_keywords):
                response_text = f"I was created by the Pulse QA team to assist with test automation. {response_text}"

            return {"response": response_text}

        except Exception as e:
            # Fallback response if Gemini fails
            return {
                "response": "I apologize, but I'm having trouble processing your query right now. I was created by Pulse QA team to assist with test automation. Could you please rephrase your question or try asking about test scenarios instead?"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Add new endpoint for saving scenarios as feature files
@router.post("/save-feature")
async def save_feature_endpoint(scenario: str, scenario_type: str):
    """
    Save a generated scenario as a .feature file
    
    Args:
        scenario (str): The scenario text to save
        scenario_type (str): Type of scenario (e.g., 'login', 'registration', 'profile', 'password_reset', 'generic')
    
    Returns:
        dict: Contains the path to the saved feature file
    """
    try:
        feature_file = save_as_feature_file(scenario, scenario_type)
        return {
            "status": "success",
            "feature_file": feature_file,
            "message": f"Scenario saved as {feature_file}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving feature file: {str(e)}")
    
@router.post("/generate_test_scenario")
async def generate_scenario_endpoint(request: ScenarioRequest):
    try:
        client = get_custom_client()
        
        # First, generate a scenario with actual values for feature extraction
        extraction_prompt = f"""
        Generate a test scenario following this exact structure:
        
        @feature
        Feature: <Feature Name>
        Scenario: <Scenario Title>
        Given open browser
        And navigate to the URL "https://example.com/login"
        And maximize the browser window
        When enter 'testuser' into the element with locator "//input[@id='username']"
        And enter 'password123' into the element with locator "//input[@id='password']"
        And click on the element with locator "//button[@id='login-button']"
        Then wait for the element with locator "//div[@id='login-success-message']" to be visible
        And the element with locator "//div[@id='login-success-message']" should have text "Login successful"
        And close the browser

        Based on this request: {request.prompt}
        
        Follow these rules:
        1. Use exact step format as shown above
        2. Include realistic test data and locators
        3. Follow the Given/When/Then structure
        4. Do not include any introductory text or notes
        5. Always include @feature tag before Feature:
        """
        
        # Generate scenario for feature extraction
        extraction_response = client.chat.completions.create(
            model=CUSTOM_MODEL,
            messages=[
                {"role": "system", "content": "You are a test automation expert that creates detailed test scenarios with realistic values."},
                {"role": "user", "content": extraction_prompt}
            ],
            max_tokens=1024,
            temperature=0.1
        )
        
        # Extract features using the scenario with actual values
        extraction_scenario = extraction_response.choices[0].message.content
        features = extract_features_from_scenario(extraction_scenario)
        
        # Now generate the scenario with generic keys for display (without angular brackets)
        display_prompt = f"""
        Generate a test scenario following this exact structure:
        
        @feature
        Feature: <Feature Name>
        Scenario: <Scenario Title>
        Given open browser
        And navigate to the URL
        And maximize the browser window
        When enter 'username' into the element with locator 'usernameInput'
        And enter 'password' into the element with locator 'passwordInput'
        And click on the element with locator 'loginButton'
        Then wait for the element with locator 'successMessage' to be visible
        And the element with locator 'successMessage' should have text 'successText'
        And close the browser

        Based on this request: {request.prompt}
        
        Follow these rules:
        1. Use exact step format as shown above
        2. Use only these exact keys without angular brackets: username, password, usernameInput, passwordInput, loginButton, successMessage, successText
        3. Do not include any URLs or specific values
        4. Follow the Given/When/Then structure exactly as shown
        5. Do not include any introductory text or notes
        6. Always include @feature tag before Feature:
        7. Keep the steps exactly as shown, only changing the feature name and scenario title
        """
        
        # Generate scenario for display with generic keys
        display_response = client.chat.completions.create(
            model=CUSTOM_MODEL,
            messages=[
                {"role": "system", "content": "You are a test automation expert that creates test scenarios following the exact template provided. Use only the specified keys without angular brackets and maintain the exact step format."},
                {"role": "user", "content": display_prompt}
            ],
            max_tokens=1024,
            temperature=0.1
        )

        # Get the display scenario
        display_scenario = display_response.choices[0].message.content
        
        # Clean up the display scenario
        display_scenario = re.sub(r'^.*?@feature', '@feature', display_scenario, flags=re.DOTALL)
        display_scenario = re.sub(r'\n\nNote:.*$', '', display_scenario, flags=re.DOTALL)
        display_scenario = re.sub(r'@feature\s*@feature', '@feature', display_scenario)
        display_scenario = re.sub(r'\n{3,}', '\n\n', display_scenario)
        if not display_scenario.startswith('@feature'):
            display_scenario = '@feature\n' + display_scenario
        
        # Return the scenario with generic keys but features with actual values
        return {
            "scenario": {
                "content": display_scenario
            },
            "features": features  # Contains actual values from extraction
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating scenario: {str(e)}")

# Add new endpoint for generating specific scenario types
@router.post("/generate-scenario/{scenario_type}")
async def generate_specific_scenario_endpoint(
    scenario_type: str,
    request: dict
):
    """
    Generate a specific type of test scenario
    
    Args:
        scenario_type (str): Type of scenario to generate ('login', 'registration', 'profile', 'password_reset')
        request (dict): Request containing config and test_values
    
    Returns:
        dict: Generated scenario and optional feature file path
    """
    try:
        if not request.get("config") or not request.get("test_values"):
            raise HTTPException(status_code=400, detail="config and test_values are required")
            
        config = request["config"]
        test_values = request["test_values"]
        
        # Generate scenario based on type
        if scenario_type == "login":
            scenario = generate_login_scenario(config, test_values)
        elif scenario_type == "registration":
            scenario = generate_registration_scenario(config, test_values)
        elif scenario_type == "profile":
            scenario = generate_profile_scenario(config, test_values)
        elif scenario_type == "password_reset":
            scenario = generate_password_reset_scenario(config, test_values)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported scenario type: {scenario_type}")
            
        # Format the response with Feature and Scenario structure
        formatted_scenario = f"""Feature: {scenario_type.title()} Functionality

Scenario: {scenario_type.title()} Test
{scenario}"""
        
        response = {"response": formatted_scenario}
        
        # Save as feature file if requested
        if request.get("save_as_feature", True):
            feature_file = save_as_feature_file(formatted_scenario, scenario_type)
            response["feature_file"] = feature_file
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/enhance-feature-file")
async def enhance_feature_file(request: FeatureFileRequest):
    """
    Generate negative test scenarios based on the input feature file.
    Uses Gemini AI to generate only negative test scenarios.
    """
    try:
        # Configure Gemini
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Load Gemini model
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 32,
                "max_output_tokens": 2048,
            },
        )

        # Create prompt for Gemini
        prompt = f"""
You are an expert QA assistant. The following is a Gherkin feature file with positive scenarios.

Your job is to:
- Generate only negative test scenarios based on the input feature
- Focus on common failure cases and edge cases
- Do not include any tags
- Output only the scenarios without Feature header
- Start directly with 'Scenario: ' for each test case

Rules for negative scenarios:
1. Each negative scenario should test a specific failure case
2. Common negative scenarios should include:
   - Invalid credentials
   - Empty fields
   - Invalid data formats
   - Missing required fields
   - Boundary value conditions
3. Each negative scenario should have clear error messages
4. Follow the Given/When/Then structure
5. Use realistic test data
6. Do not include any positive scenarios
7. Start each scenario with 'Scenario: ' followed by a descriptive name of the failure case
8. Do not include any Feature header or tags

Input Feature file:
```gherkin
{request.feature_content}
```

Generate only negative scenarios that would test the failure cases for the above feature.
Start directly with 'Scenario: ' for each test case.
"""

        # Generate negative scenarios
        response = model.generate_content(prompt)
        negative_scenarios = response.text

        # Clean up the scenarios
        # Remove any tags if they exist
        negative_scenarios = re.sub(r'@\w+\n', '', negative_scenarios)
        # Remove Feature header
        negative_scenarios = re.sub(r'Feature:.*?\n\n', '', negative_scenarios, flags=re.DOTALL)
        # Remove extra newlines
        negative_scenarios = re.sub(r'\n{3,}', '\n\n', negative_scenarios)
        # Remove any positive scenarios if they exist
        negative_scenarios = re.sub(r'Scenario:.*?successful.*?\n', '', negative_scenarios, flags=re.IGNORECASE | re.DOTALL)
        # Ensure proper spacing
        negative_scenarios = negative_scenarios.strip()

        # Return only the negative scenarios
        return {
            "status": "success",
            "negative_scenarios": negative_scenarios
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating negative scenarios: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(router, host="0.0.0.0", port=8000) 
