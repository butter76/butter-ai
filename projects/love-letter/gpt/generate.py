import torch
import yaml
from gpt.models.gpt import GPTModel
from gpt.models.tokenizer import LoveLetterTokenizer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate():
    # Load configuration
    config_path = 'gpt/config/model_config.yaml'
    config = load_config(config_path)
    model_config = config['model']
    seq_length = model_config['seq_length']
    
    # Set device
    device = torch.device(model_config['device'] if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer and model
    tokenizer = LoveLetterTokenizer()
    model = GPTModel(model_config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load('checkpoints/model_epoch_1.pt')  # Adjust epoch number as needed
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Starting prompt
    prompt = "|gamestart\n|p1|hidden|draw|5\n|p2|hidden|draw|1\n|turn|p1\n|p1|hidden|draw|1\n|yourmove|p1\n|p1|play|1|6\n|turn|p2\n|p2|hidden|draw|4"
    tokens = tokenizer.tokenize(prompt)


    with torch.no_grad():
        output_tokens = model.generate(
            tokens,
            max_new_tokens=30,
        )    
    # Convert back to text
    generated_text = tokenizer.detokenize(output_tokens)
    print("Generated game log:")
    print(generated_text)

if __name__ == '__main__':
    generate()
