import json
from time import sleep

def create_eval_ds(agent, ground_truth_path):
    test_data = []
    
    with open(ground_truth_path, 'r') as f:
        data = json.load(f)

    for obj in data:
        print(f'Question: {obj["question"]}')
        resp = {
            'user_input': obj["question"],
            'reference': obj["ground_truth"],
            'retrieved_contexts': [],
            'response': None
        }

        # Get agent response
        try:
            response = agent.run(obj["question"], markdown=True)
            resp['response'] = response.content
            
            # Safely handle references
            if hasattr(response, 'extra_data') and response.extra_data is not None:
                if hasattr(response.extra_data, 'references'):
                    relevant_docs = []
                    for reference in response.extra_data.references:
                        if hasattr(reference, 'references'):
                            relevant_docs.extend([
                                item['content'] 
                                for item in reference.references 
                                if 'content' in item
                            ])
                    resp['retrieved_contexts'] = relevant_docs
            
            test_data.append(resp)
            print(f"Response: {response.content[:200]}...")  # Truncated for display
            if resp['retrieved_contexts']:
                print(f"Found {len(resp['retrieved_contexts'])} relevant documents")
            
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            test_data.append(resp)  # Store what we have
        
        sleep(1)  # Reduced from 60s to avoid unnecessary delays
    
    return test_data