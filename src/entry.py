import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.rnn.rnn import RNN
from src.rnn.train_rnn import train

import torch
import src.rnn.sequence_engine as sequence_engine

def main():
	seq_length = 12
	epochs = 300

	x_data, y_data = sequence_engine.create_sequences(
		sequence_engine.passengers_normalized,
		seq_length,
	)

	x = torch.tensor(x_data, dtype=torch.float32).unsqueeze(-1)
	y = torch.tensor(y_data, dtype=torch.float32).unsqueeze(-1)

	model = RNN(input_size=1, hidden_size=32, output_size=1)
	model = train(None, model, epochs, x, y)
	model.eval()

	with torch.no_grad():
		training_loss = torch.mean((model(x) - y) ** 2).item()

		last_sequence = torch.tensor(
			sequence_engine.passengers_normalized[-seq_length:],
			dtype=torch.float32,
		).unsqueeze(0).unsqueeze(-1)

		next_normalized = model(last_sequence).item()

	next_normalized = min(max(next_normalized, 0.0), 1.0)
	next_passengers = (
		next_normalized
		* (sequence_engine.passengers_max - sequence_engine.passengers_min)
		+ sequence_engine.passengers_min
	)

	print(f"Trained on {len(x_data)} sequences")
	print(f"Final training MSE: {training_loss:.6f}")
	print(f"Predicted next month passengers: {next_passengers:.2f}")


if __name__ == "__main__":
	main()




