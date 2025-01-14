import torch
from gpt.models.gpt_ll import LoveLetterTransformer
from gpt.models.tokenizer import LoveLetterTokenizer
from gpt.models.cli import get_generate_parser, load_config, update_config_with_args

def generate(args):
    # Load and update config
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    model_config = config['model']
    generation_config = config['generation']
    
    # Set device
    device = torch.device(model_config['device'] if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer and model
    tokenizer = LoveLetterTokenizer()
    model = LoveLetterTransformer(
        vocab_size=tokenizer.vocab_size,
        model_config=model_config,
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = generation_config['checkpoint_path']
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get tokens from prompt
    tokens = tokenizer.tokenize(generation_config['prompt'])

    with torch.no_grad():
        output_tokens = model.generate(
            tokens,
            max_new_tokens=generation_config['max_tokens'],
            temperature=generation_config['temperature'],
        )    
    # Convert back to text
    generated_text = tokenizer.detokenize(output_tokens)
    print("Generated game log:")
    print(generated_text)

if __name__ == '__main__':
    parser = get_generate_parser()
    args = parser.parse_args()
    generate(args)

