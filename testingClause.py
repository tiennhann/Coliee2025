import dspy
import json
import os
import joblib
import dill
from dspy.teleprompt import BootstrapFewShot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration for DSPy
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
ollama_model = dspy.OllamaLocal(model="llama2", model_type='text',
                                max_tokens=1000,
                                temperature=0.1,
                                top_p=0.9,
                                frequency_penalty=0.0,
                                top_k=50)

# Configure DSPy settings
dspy.settings.configure(rm=colbertv2_wiki17_abstracts, lm=ollama_model)

class ContractDetails(dspy.Signature):
    """Determine if a clause is consistent with the contract text."""
    contract = dspy.InputField(desc="Full text of the contract")
    clause = dspy.InputField(desc="A single clause to be verified")
    question = dspy.InputField(desc="Question about the clause")
    answer = dspy.OutputField(desc="Answer 'yes' if the clause's content is mentioned in or consistent with the contract text, 'no' if it's not mentioned or contradicts the contract.")
    reasoning = dspy.OutputField(desc="Detailed step-by-step reasoning for the answer, referencing specific parts of the contract text")
    confidence = dspy.OutputField(desc="Confidence score between 0 and 1, where 1 is highest confidence")

class GenerateContractDetails(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_details = dspy.ChainOfThought(ContractDetails)

    def forward(self, contract, clause, question):
        instructions = """
        When evaluating the clause:
        1. Check if the main ideas or information in the clause are mentioned in the contract.
        2. Consider that "at or above" a certain grade or quality means that higher grades are acceptable.
        3. Remember that all activities are to be completed within one month, which is equivalent to 4 weeks.
        4. If the clause provides more specific details but doesn't contradict the contract, it may still be consistent.
        5. Answer 'yes' if the clause information is mentioned in or consistent with the contract, 'no' if it's not mentioned or contradicts the contract.
        6. Provide a clear reasoning that references specific parts of the contract.
        7. Provide a confidence score between 0 and 1, where 1 is highest confidence.

        Format your response as:
        Answer: [yes/no]
        Reasoning: [Your step-by-step reasoning]
        Confidence: [0-1]
        """
        return self.generate_details(contract=contract, clause=clause, question=question + "\n" + instructions)

def semantic_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def validate_contract_details(example, pred, trace=None):
    valid_answers = ['yes', 'no']
    if pred.answer.lower() not in valid_answers:
        return 0

    contract_text = example.contract.lower()
    clause_text = example.clause.lower()

    # Calculate the similarity score
    similarity_score = semantic_similarity(contract_text, clause_text)
    #print(similarity_score)
    
    # Calculate match ratio for key terms
    key_elements = clause_text.split()
    matches = sum(1 for element in key_elements if element in contract_text)
    match_ratio = matches / len(key_elements)

    # Combined score for general consistency
    combined_score = (similarity_score + match_ratio) / 2

    # Validate based on combined score
    if combined_score > 0.6 and pred.answer.lower() == 'yes':
        return 1
    elif combined_score <= 0.6 and pred.answer.lower() == 'no':
        return 1
    else:
        return 0

def load_training_data(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    train = []
    for contract in json_data['contracts']:
        contract_text = contract['contract_text']
        question = contract['question']
        for clause in contract['clauses']:
            train.append(
                dspy.Example(
                    contract=contract_text, 
                    clause=clause['clause'], 
                    question=question, 
                    answer=clause['answer']
                ).with_inputs('contract', 'clause', 'question')
            )
    return train

def save_model_data(compiled_model, filename):
    try: 
        model_data = {
            # Create a dictionary with all necessary data
            'config': {
                'ollama': {
                    'model': "llama3",
                    'model_type': 'text',
                    'max_tokens': 1000,
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'frequency_penalty': 0.0,
                    'top_k': 50
                },
                'colbert': {
                    'url': dspy.settings.rm.url
                }
            },
            'signature': {
                'input_fields': ['contract', 'clause', 'question'],
                'output_fields': ['answer', 'reasoning', 'confidence'],
                'instructions': 'Determine if a clause is consistent with the contract text.'
            },
            'training_data': getattr(compiled_model, '_training_data', None),
            'bootstrap_examples': getattr(compiled_model, '_bootstrap_examples', None)      # Save the entire model
        }

        with open(filename, 'wb') as f:
            joblib.dump(model_data, f)
        print(f"Model saved successfully to {filename}")
    except Exception as e:
        print(f"Error while saving: {str(e)}")
        raise

def load_model_data(filename):
    try:
        with open(filename, 'rb') as f:
            model_data = joblib.load(f)
        
        # Reconstruct the model configuration
        lm = dspy.OllamaLocal(**model_data['config']['ollama'])
        rm = dspy.ColBERTv2(url=model_data['config']['colbert'])
        
        # Configure DSPy settings
        dspy.settings.configure(lm=lm, rm=rm)
        
        # Create and compile a new model with the saved configuration
        model = GenerateContractDetails()

        # If we have training state, apply it
        if model_data.get('training_state'):
            if hasattr(model, 'predictor'):
                model.predictor.config = model_data['training_state'].get('predictor_config')
            model.examples = model_data['training_state'].get('examples', [])
            model.metric_config = model_data['training_state'].get('metric_config')
        
        return model
    except Exception as e:
        print(f"Error while loading: {str(e)}")
        raise

def main():
    #model_file = 'compiled_contract_details.dill'
    model_file = 'compiled_contract_details.joblib'
    try:
        # Load the compiled model if it exists
        if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
            print("Loading pre-trained model..")
            #with open(model_file, 'rb') as file:
            compiled_contract_details = load_model_data(model_file)
            print("Model loaded successfully.")
        else:
            raise FileNotFoundError("Model file not found or empty.")
    except Exception as e:
            # Train the model if it doesn't exist
            print(f"Error loading model: {str(e)}")
            print("Training new model....")
            train = load_training_data('output_json.json')
            teleprompter = BootstrapFewShot(metric=validate_contract_details)
            compiled_contract_details = teleprompter.compile(GenerateContractDetails(), trainset=train)

            # Save the compiled model
            print("Saving model configuration....")
            try:
                #with open(model_file, 'wb') as file:
                save_model_data(compiled_contract_details, model_file)
                print("Model saved successfully.")
            except Exception as save_error:
                print(f"Error saving model: {str(save_error)}")
                print("Proceeding with unsaved model.")

    new_contract_text = """
    (A Contract Between XYZ Homes and Lumber Yard A). XYZ Homes builds eight to nine 2,000 square foot homes each month for new home buyers. Each new home requires 16,000 board feet of Number 2 Common grade lumber. In order to complete eight to nine homes, XYZ Homes must purchase 144,000 board feet of Number 2 Common grade lumber each month. Lumber Yard A is the preferred supplier of this lumber.\nIn the first part of our scenario, we look at the agreement that XYZ Homes contracts with Lumber Yard A for the required lumber. The agreement specifies the responsibilities of each agent. It is formalized as a set of constraints on how the work is to be conducted. These constraints can be viewed as requirements (in the sense of the CPS Framework), and each requirement is mapped to one or more of an agent's concerns. A sample of these constraints and concerns includes:\n1. Lumber Yard A will produce a total of 144,000 board feet of lumber for XYZ Homes. This constraint addresses the functionality concern of XYZ Homes.\n2. Lumber Yard A guarantees to schedule the transport and delivery of 14-16 tractor trailers worth of lumber in one month to XYZ Homes. This constraint addresses the time to market concern.\n3. The lumber delivered to XYZ Homes will be at or above Number 2 Common grade.This constraint addresses several concerns including physical, reliability, quality and trustworthiness. For example, if Lumber Yard A were to provide lumber that is of a lesser quality than Number 2 Common grade, then from XYZ Home's perspective, Lumber Yard A would no longer be trustworthy.\n4. The agreed upon cost of lumber is at $122,000 for 144,000 board feet and the transport and delivery cost will be at or below $500,000 for 144,000 board feet. This constraint addresses the cost concern.
    """

    question = "Is the information in this clause mentioned in or consistent with the content of the contract, considering that all activities are to be completed within one month? Answer 'yes' if the clause information is mentioned in or consistent with the contract, 'no' if it's not mentioned or contradicts the contract."

    clauses = [
        "C1: Lumber Yard A responsible for producing Number 3 Common grade lumber when by week 4.",
        "C2: Lumber Yard A responsible for delivering 100,000 board feet of Number 2 Common or above grade lumber to XYZ Homes when by week 4.",
        "C3: Lumber Yard A responsible for ensuring the delivered lumber meets Number 2 Common or above quality standards when by week 4.",
        "C4: Lumber Yard A responsible for scheduling transport and delivery of 13-16 tractor trailers worth of lumber to XYZ Homes within one month from contract signing.",
        "C5: XYZ Homes responsible for paying $90,000 for the 144,000 board feet of Number 2 Common or above grade lumber when payment is due.",
        "C6: Lumber Yard A responsible for ensuring transport and delivery cost does not exceed $500,000 for the 122,000 board feet of Number 2 Common or above grade lumber."
    ]

    results = []
    for clause in clauses:
        pred = compiled_contract_details(contract=new_contract_text, clause=clause, question=question)
        results.append({
            'clause': clause,
            'answer': pred.answer,
            'reasoning': pred.reasoning,
            'confidence': pred.confidence
        })

    for result in results:
        print(f"Clause: {result['clause']}")
        print(f"Answer: {result['answer']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Confidence: {result['confidence']}")
        print("---")

if __name__ == "__main__":
    main()
    