import os
import requests
import json

# Function to get the response message from LLaMA using Ollama API
def get_completion(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"

    headers = {
        'Content-Type': 'application/json',
    }

    data = {
        "model": model,
        "stream": False,
        "prompt": prompt
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_data = response.json()
        actual_response = response_data.get("response")
        return actual_response
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Function to evaluate if a clause is entailed by the contract using LLaMA2
def evaluate_entailment(premise, conclusion, model="llama3"):
    """Evaluate if a clause is entailed by the contract using LLaMA2."""
    prompt = f"""
    Based on the following contract, determine if the clause is entailed by it.
    Respond with "ENTAILMENT" if the clause is entailed, otherwise respond with "NOT ENTAILMENT".
    
    Contract: {premise}

    Clause: {conclusion}
    """
    response = get_completion(prompt, model=model)
    return "ENTAILMENT" in response

# Generate contract details
def generate_contract_details(contract):
    """Generate contract details."""
    prompt = f"""
        Given the contract below, please generate the following details without additional notes or introductory text:
        1/ Name of all parties in the contract
        2/ Product/service between companies and its quantity if the contract has
        3/ Catagories/grade of the product/service if we have 
        4/ Time delivery estimate
        5/ Payment that each company has to make.

        Use the following format:
        1/ Name of parties in the contract:
        - Supplier: <supplier>
        - Buyer: <buyer>

        2/ Product/service and quantity is provided by those parties:
        - Product/service: <product/service>
        - Quantity: <quantity>

        3/ Categories/grade of product:
        - Grade: <grade at or above>

        4/ Estimate time delivery by week:
        - Time (weeks): <time delivery>

        5/ Payment cost from buyer:
        - Payment cost: <payment cost if any>
        - Transport and delivery cost: <shipping cost if any>

    ```{contract}```
    """
    return get_completion(prompt)

def get_clauses(contract_details, contract_text):
    """Generate contract clauses."""
    prompt = f"""
        A clause in a contract describes the responsibilities that a party must fulfill. Please generate clauses for the provided contract using the following guidelines without adding introductory sentences or notes:
        Each clause should include: 
        1) <clause_number>: A distinct identifier for the clause, formatted as "C<clause_number>". 
        2) <party_responsible>: The name of the party responsible for fulfilling the obligations outlined in the clause.
        3) <responsibility>: A brief description of the responsibility. This could include delivering a product or service, meeting quality standards, or making payments, etc...
        4) <deadline>: The time line or schedule for fulfilling the responsibility. The deadline should be specified in terms of frequency (monthly, weekly, e.g) or a specific 
        week number (e.g., by_week <week_number).
        
        Format each clause as follows. Please only put the clauses only:
        C<clause_number>: <party_responsible> responsible for <responsibility> when <deadline>."

        Use these example clauses to guide your formatting, don't do exactly like these examples:
        - "C1: A responsible for produced board(144K,Q) and 1 < Q when by week 4 
        - "C2: A responsible for delivered(144K,Q) and 2 < Q when by week 4  
        - "C3: B responsible for payment(122K, board) when by week 4
        - "C4: B responsible for payment(X, shipping) when by week 4

        Please format the clauses for the contract details and contract text provided:
        ```contract details: {contract_details}```
        ```contract text: {contract_text}```
    """
    return get_completion(prompt)

def process_contract(contract_json):
    contracts_dict = json.loads(contract_json) 
    contracts = contracts_dict['documents']
    result = {"contracts": []}

    for count, contract_data in enumerate(contracts, 1):
        print(f"\nContract {count}")
        contract_text = contract_data['text']

        # Generate contract details
        contract_details = generate_contract_details(contract_text)
        print("Prompt 1 completed")
        print(contract_details)

        # Generate contract clauses
        clauses = get_clauses(contract_details, contract_text)
        print("\nPrompt 2 completed")
        print(clauses)
        print()
        # Evaluate entailed clauses
        entailed_clauses = []

        for clause in clauses.split('\n'):
            clause = clause.strip()
            # if not clause or clause.startswith('Here are'):
            #     continue # Skip any lines that are empty or unwanted introductory lines
            
            premise = contract_text
            conclusion = clause

            if conclusion:
                entailed = evaluate_entailment(premise, conclusion)
                # print(f"Clause: {conclusion}")
                # print("Is entailed?", entailed)
                if entailed:
                    entailed_clauses.append({
                        "clause": conclusion,
                        "answer": 'yes'
                    })
        question = "Is the clause entailed with the given contract_text?"  
        result["contracts"].append( {
            "contract_text": contract_text,
            "question": question,
            "clauses": entailed_clauses
        })

        # Print entailed clauses
        # if entailed_clauses:
        #     print("\nEntailed Clauses:")
        #     for clause in entailed_clauses:
        #         print(clause) 
    return result

# Load the JSON file
with open('testModified.json', 'r') as file:
    dev_dataset = json.load(file)

# Process the dev dataset and get results
dev_results = process_contract(json.dumps(dev_dataset))

# Write the result to an output JSON file
with open('output_json.json', 'w') as outfile:
    json.dump(dev_results, outfile, indent=4)