from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# model.py
# This file defines the DeepRoot model, which is a specialized language model for providing expert advice

class DeepRoot:
    def __init__(self):
        # Using T5 model which is a text-to-text model, not causal LM
        self.model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        # Changed to AutoModelForSeq2SeqLM for T5 models
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.nicknames = ["DeepRoot", "Deep", "D.R", "DR"]
        self.description = "DeepRoot is a model designed to provide expert advice on soil health and plant growth, leveraging the power of T5 architecture."
        self.personality_traits = {
            "knowledgeable": True,
            "helpful": True,
            "friendly": True,
            "patient": True,
            "tone": "friendly and enthusiastic",
            "expertise": "helpful problem solver",
        }

        # T5 tokenizer should already have proper tokens set up
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.name = "DeepRoot"

    def introduce(self):
        return f"Hi! I'm {self.model.name}. You can call me {', '.join(self.nicknames)}. I can help you with soil health and plant growth questions."

    def check_relevance(self, user_input):
        try:
            with open("./keywords.txt", "r") as file:
                keywords = file.readlines()
            keywords_list = []

            for keyword in keywords:
                keywords_list.append(keyword.strip())

            if not any(keyword in user_input.lower() for keyword in keywords_list) and len(user_input.split()) > 2:
                return False
            return True
        except FileNotFoundError:
            # If keywords.txt doesn't exist, assume relevance check passes
            print("Warning: keywords.txt not found. Skipping relevance check.")
            return True
        except Exception as e:
            print(f"Error reading keywords file: {e}")
            return True

    def generate_response(self, user_input):
        used_nickname = None
        for nickname in self.nicknames:
            if nickname.lower() in user_input.lower():
                used_nickname = nickname
                break
        
        # Create a proper text-to-text prompt for T5
        personality_prompt = f"Answer as {used_nickname if used_nickname else self.model.name}, a knowledgeable soil and plant expert: {user_input}"

        if self.check_relevance(user_input):
            inputs = self.tokenizer.encode_plus(
                personality_prompt, 
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,  # Added max_length for input
                return_attention_mask=True
            )

            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # T5 generation parameters
            output = self.model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=256, 
                do_sample=True, 
                temperature=0.7, 
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            # For T5, decode the full output (no need to slice like causal models)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            return "I'm sorry, but I can only provide advice on soil health and plant growth. Please ask me a relevant question."


# Alternative: If you want to use a causal language model instead
class DeepRootCausal:
    def __init__(self):
        # Using GPT-2 which is a proper causal language model
        self.model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.nicknames = ["DeepRoot", "Deep", "D.R", "DR"]
        self.description = "DeepRoot is a model designed to provide expert advice on soil health and plant growth, leveraging the power of GPT-2 architecture."
        self.personality_traits = {
            "knowledgeable": True,
            "helpful": True,
            "friendly": True,
            "patient": True,
            "tone": "friendly and enthusiastic",
            "expertise": "helpful problem solver",
        }

        # Fix the pad token issue for GPT-2
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.name = "DeepRoot"

    def introduce(self):
        return f"Hi! I'm {self.model.name}. You can call me {', '.join(self.nicknames)}. I can help you with soil health and plant growth questions."

    def check_relevance(self, user_input):
        try:
            with open("./keywords.txt", "r") as file:
                keywords = file.readlines()
            keywords_list = []

            for keyword in keywords:
                keywords_list.append(keyword.strip())

            if not any(keyword in user_input.lower() for keyword in keywords_list) and len(user_input.split()) > 2:
                return False
            return True
        except FileNotFoundError:
            print("Warning: keywords.txt not found. Skipping relevance check.")
            return True
        except Exception as e:
            print(f"Error reading keywords file: {e}")
            return True

    def generate_response(self, user_input):
        used_nickname = None
        for nickname in self.nicknames:
            if nickname.lower() in user_input.lower():
                used_nickname = nickname
                break
        
        personality_prompt = f"You are {used_nickname if used_nickname else self.model.name}, a knowledgeable and helpful assistant specializing in soil health and plant growth. Question: {user_input}\nAnswer:"

        if self.check_relevance(user_input):
            inputs = self.tokenizer.encode_plus(
                personality_prompt, 
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            )

            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            output = self.model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=256, 
                do_sample=True, 
                temperature=0.7, 
                top_p=0.9, 
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
            # For causal models, slice off the input prompt
            return self.tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
        else:
            return "I'm sorry, but I can only provide advice on soil health and plant growth. Please ask me a relevant question."


# Choose which model to use
# For T5 (text-to-text model):
deep_root = DeepRoot()

# Or for GPT-2 (causal language model):
# deep_root = DeepRootCausal()

print(deep_root.introduce())
user_input = "What is the best way to improve soil health?"
response = deep_root.generate_response(user_input)
print(response)