import json
from time import sleep
import re  # For regex operations

def remove_thinking_steps(response_content):
    """
    Remove <think>...</think> sections from the response content using regex.
    
    Args:
        response_content (str): The raw response content from the agent.
    
    Returns:
        str: The cleaned response content without <think> sections.
    """
    return re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()

def create_eval_ds(agent, ground_truth_path):
    """
    Create an evaluation dataset by running the agent on questions from the ground truth file.
    
    Args:
        agent: The RAG agent to evaluate.
        ground_truth_path (str): Path to the JSON file containing ground truth data.
    
    Returns:
        list: A list of dictionaries containing evaluation data.
    """
    test_data = []
    
    # Load the ground truth data
    try:
        with open(ground_truth_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return test_data

    for obj in data:
        print(f'Question: {obj["question"]}')
        
        # Initialize the response dictionary
        resp = {
            'user_input': obj["question"],
            'reference': obj["ground_truth"],
            'retrieved_contexts': [],
            'response': None
        }

        # Get the agent's response
        try:
            response = agent.run(obj["question"], markdown=True)
            
            # Clean the response content by removing <think> sections
            cleaned_response = remove_thinking_steps(response.content)
            resp['response'] = cleaned_response
            
            # Safely handle references (if they exist)
            if hasattr(response, 'extra_data') and response.extra_data is not None:
                if 'references' in response.extra_data:
                    relevant_docs = []
                    for reference in response.extra_data['references']:
                        if 'content' in reference:
                            relevant_docs.append(reference['content'])
                    resp['retrieved_contexts'] = relevant_docs
                else:
                    print("No references found in extra_data.")
            else:
                print("No extra_data field in the response.")
            
            # Append the result to the test data
            test_data.append(resp)
            
            # Print a truncated version of the response for debugging
            print(f"Cleaned Response: {cleaned_response[:200]}...")  # Truncated for display
            if resp['retrieved_contexts']:
                print(f"Found {len(resp['retrieved_contexts'])} relevant documents")
        
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            test_data.append(resp)  # Store what we have, even if incomplete
        
        # Sleep to avoid overwhelming the agent or API rate limits
        sleep(1)  # Reduced from 60s to avoid unnecessary delays
    
    return test_data