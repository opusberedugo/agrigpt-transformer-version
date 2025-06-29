from flask import Flask, request, jsonify
from model import DeepRoot

app = Flask(__name__)
deeproot = DeepRoot()

# @app.route('/introduce', methods=['GET'])
# def introduce():
    # introduction = deep_root.introduce()
    # return jsonify({"introduction": introduction})

@app.route('/chat', methods=['POST', 'GET'])
def chat():
  message = request.args.get('message')
  print(f"Received message: {message}")
  if not message:
    return jsonify({"error": "No message provided"}), 400
  else:
     return jsonify({"message": f"{deeproot.generate_response(message)}"})
  # return jsonify({"message": f" {message} "})

if __name__ == '__main__':
    app.run(debug=True)
