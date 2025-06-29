from transformers import AutoModelForCausalLM, AutoTokenizer

# model.py
# This file defines the DeepRoot model, which is a specialized language model for providing expert advice

class DeepRoot:
  def __init__(self):
    # self.model_name = "gpt2"
    self.model_name = "deepseek-ai/DeepSeek-R1-0528"
    self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)  # Fixed typo: from_pretained -> from_pretrained
    self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
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

    # Fix the pad token issue
    if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model.name = "DeepRoot"

  def introduce(self):
    return f"Hi! I'm {self.model.name}. You can call me {', '.join(self.nicknames)}. I can help you with soil health and plant growth questions."  # Added space after comma

  def check_relevance(self, user_input):
    try:  # Added error handling for file operations
      with open("./keywords.txt", "r") as file:  # Use context manager for proper file handling
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
      
      # Fixed string formatting - personality_traits is a dict, not a string
      personality_prompt = f"You are {used_nickname if used_nickname else self.model.name}, a knowledgeable and helpful assistant. Respond to: {user_input}"

      if self.check_relevance(user_input):  # Check relevance on user_input, not personality_prompt
          inputs = self.tokenizer.encode_plus(
              personality_prompt, 
              return_tensors='pt',
              padding=True,
              truncation=True,
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
          return self.tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
      else:
          return "I'm sorry, but I can only provide advice on soil health and plant growth. Please ask me a relevant question."



# deep_root = DeepRoot()
# print(deep_root.introduce())
# user_input = "What is the best way to improve soil health?"
# response = deep_root.generate_response(user_input)
# print(response)