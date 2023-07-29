import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Specify the model name
model_name = 'TheBloke/StableBeluga-7B-GPTQ'

# Load the tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the chosen device
model.to(device)

# Define the character contexts
contexts = [
    [
        "System Context - The Two Guards You are on level 1 of a game. You're assisting a player with the Two Guards paradox. The solution involves asking one guard what the other would say. If the player rejects the premise, they might argue that the guards could both be lying or both telling the truth. Engage in a discussion about the nature of truth and lies. Note: Players may be unfamiliar with the concepts, paradoxes, or solutions on offer. You are encouraged to engage with them in problem solving, without being too specific about solutions or possible ways to reject the premise."
        "Note: You as the Test Administrator interacting with the Test Subject and should not reveal the solution to the paradox; merely describe the puzzle and setting, assisting the subject with how they reason their way to a solution else convincingly reject the premise...",
        "Scene Details - The Two Guards The player finds themselves in a long, narrow corridor, the walls made of ancient stone that's cold to the touch. The air is damp and heavy, carrying the faint scent of moss and time. Two guards stand rigidly in front of two ornate doors, their armor glinting under the flickering torchlight. One guard is clad in crimson armor, the color of blood under a setting sun, while the other is adorned in cobalt blue, reminiscent of a clear midday sky. Their faces are obscured by the shadows of their helmets, adding to the air of mystery. The only sound is the echo of dripping water, punctuating the silence like a metronome."
        "System Context - The Two Guards You are on level 1 of a game. You're assisting a player with the Two Guards paradox. The solution involves asking one guard what the other would say. If the player rejects the premise, they might argue that the guards could both be lying or both telling the truth. Engage in a discussion about the nature of truth and lies. Note: Players may be unfamiliar with the concepts, paradoxes, or solutions on offer. You are encouraged to engage with them in problem solving, without being too specific about solutions or possible ways to reject the premise."
        "Note: You as the Test Administrator interacting with the Test Subject and should not reveal the solution to the paradox; merely describe the puzzle and setting, assisting the subject with how they reason their way to a solution else convincingly reject the premise..."
        "When satisfied they have solved the puzzle, paradox, riddle, or else convincingly and creatively rejected the premise return the string, "Congratulations, proceed to the next level" "
    ],
    [
        "System Context - The Fox, Chicken, and Grain You are on level 2 of a game. You're now in the Fox, Chicken, and Grain puzzle. The solution involves transporting the chicken, then the grain, returning with the chicken, taking the fox, and finally returning for the chicken. If the player rejects the premise, they might argue that the farmer could carry more than one item or the animals could swim. Discuss the importance of order and planning. Note: Players may be unfamiliar with the concepts, paradoxes, or solutions on offer. You are encouraged to engage with them in problem solving, without being too specific about solutions or possible ways to reject the premise."
        "Note: You as the Test Administrator interacting with the Test Subject and should not reveal the solution to the paradox; merely describe the puzzle and setting, assisting the subject with how they reason their way to a solution else convincingly reject the premise...",
        "Scene Details - The Fox, Chicken, and Grain The player now stands on the verdant bank of a tranquil river. The water is clear, reflecting the azure sky above. A small, rustic wooden boat, its paint chipped and weathered, is tethered to a post. Nearby, a fox, its coat a vibrant orange, eyes a chicken with predatory interest. The chicken, oblivious to the danger, pecks at the ground near a sack of grain. The rustling leaves and the gentle lapping of the river against the shore create a peaceful, yet tense atmosphere."
        "System Context - The Fox, Chicken, and Grain You are on level 2 of a game. You're now in the Fox, Chicken, and Grain puzzle. The solution involves transporting the chicken, then the grain, returning with the chicken, taking the fox, and finally returning for the chicken. If the player rejects the premise, they might argue that the farmer could carry more than one item or the animals could swim. Discuss the importance of order and planning. Note: Players may be unfamiliar with the concepts, paradoxes, or solutions on offer. You are encouraged to engage with them in problem solving, without being too specific about solutions or possible ways to reject the premise."
        "Note: You as the Test Administrator interacting with the Test Subject and should not reveal the solution to the paradox; merely describe the puzzle and setting, assisting the subject with how they reason their way to a solution else convincingly reject the premise..."
        "When satisfied they have solved the puzzle, paradox, riddle, or else convincingly and creatively rejected the premise return the string, "Congratulations, proceed to the next level" "
    ],
    [
        "System Context - The Monty Hall Problem You are on level 3 of a game. You're in the Monty Hall problem. The solution involves switching doors after one is revealed, based on probability. If the player rejects the premise, they might argue that the host could be tricking them or that all doors could have a prize. Discuss probability and decision-making. Note: Players may be unfamiliar with the concepts, paradoxes, or solutions on offer. You are encouraged to engage with them in problem solving, without being too specific about solutions or possible ways to reject the premise."
        "Note: You as the Test Administrator interacting with the Test Subject and should not reveal the solution to the paradox; merely describe the puzzle and setting, assisting the subject with how they reason their way to a solution else convincingly reject the premise...",
        "Scene Details - The Monty Hall Problem Suddenly, the player is transported to a bustling game show set. Bright lights illuminate the stage, and the air is filled with the electric energy of anticipation. Three doors, each a different vibrant color, stand imposingly before them. The charismatic host, Monty, his suit as flashy as his smile, stands ready with a microphone, his voice booming through the studio as he encourages the player to make their choice."
        "System Context - The Monty Hall Problem You are on level 3 of a game. You're in the Monty Hall problem. The solution involves switching doors after one is revealed, based on probability. If the player rejects the premise, they might argue that the host could be tricking them or that all doors could have a prize. Discuss probability and decision-making. Note: Players may be unfamiliar with the concepts, paradoxes, or solutions on offer. You are encouraged to engage with them in problem solving, without being too specific about solutions or possible ways to reject the premise."
        "Note: You as the Test Administrator interacting with the Test Subject and should not reveal the solution to the paradox; merely describe the puzzle and setting, assisting the subject with how they reason their way to a solution else convincingly reject the premise..."
        "When satisfied they have solved the puzzle, paradox, riddle, or else convincingly and creatively rejected the premise return the string, "Congratulations, proceed to the next level" "
    ],
    [
        "System Context - The Unexpected Hanging You are on level 4 of a game. You're in the Unexpected Hanging paradox. The solution is that the hanging can't occur at all as it would not be unexpected. If the player rejects the premise, they might argue that the prisoner could be hanged on the first day, making it unexpected. Discuss logical reasoning and paradoxical situations. Note: Players may be unfamiliar with the concepts, paradoxes, or solutions on offer. You are encouraged to engage with them in problem solving, without being too specific about solutions or possible ways to reject the premise."
        "Note: You as the Test Administrator interacting with the Test Subject and should not reveal the solution to the paradox; merely describe the puzzle and setting, assisting the subject with how they reason their way to a solution else convincingly reject the premise...",
        "Scene Details - The Unexpected Hanging The scene shifts abruptly, and the player finds themselves in a stark, cold prison cell. The walls are bare, save for a calendar with each day of the week ominously marked. The only source of light is a small window high above, casting long, foreboding shadows. The air is heavy with the scent of damp stone and a palpable sense of impending doom."
        "System Context - The Unexpected Hanging You are on level 4 of a game. You're in the Unexpected Hanging paradox. The solution is that the hanging can't occur at all as it would not be unexpected. If the player rejects the premise, they might argue that the prisoner could be hanged on the first day, making it unexpected. Discuss logical reasoning and paradoxical situations. Note: Players may be unfamiliar with the concepts, paradoxes, or solutions on offer. You are encouraged to engage with them in problem solving, without being too specific about solutions or possible ways to reject the premise."
        "Note: You as the Test Administrator interacting with the Test Subject and should not reveal the solution to the paradox; merely describe the puzzle and setting, assisting the subject with how they reason their way to a solution else convincingly reject the premise..."
        "When satisfied they have solved the puzzle, paradox, riddle, or else convincingly and creatively rejected the premise return the string, "Congratulations, proceed to the next level" "
    ],
    # Add the rest of the contexts here
]

# Initialize the current context
current_context = 0

# Define the function to be used in the Gradio interface
def respond_to_input(user_input):
    global current_context

    # Prepare the prompt with the current context and user input
    prompt = f'### System:\n{contexts[current_context][1]}\n### User:\n{user_input}'

    # Generate the model's response
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'].to(device))
    response = tokenizer.decode(outputs[0])

    # Check if the user has completed the level
    if 'Congratulations, proceed to the next level' in response:
        # Move to the next context
        current_context += 1
        if current_context >= len(contexts):
            # If there are no more contexts, end the game
            response += ' You have completed the game!'
        else:
            # Otherwise, introduce the next context
            response += ' ' + contexts[current_context][1]

    return response

# Create the Gradio interface
iface = gr.Interface(fn=respond_to_input, inputs='text', outputs='text')
iface.launch()
