from models import MLP

mlp = MLP(num_genes=36601, hidden1=2048, hidden2=512)  # Initialize the model
mlp.load_state_dict(torch.load('./mlp_model.pth'))  # Load the saved model
mlp.eval()
all_labels = []
all_predictions = []
with torch.no_grad():
  for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = mlp(inputs)
    _, predicted = torch.max(outputs.data, 1)
    all_labels.extend(labels.cpu().numpy())
    all_predictions.extend(predicted.cpu().numpy())

    # Calculate F1 Score
f1 = f1_score(all_labels, all_predictions, average='binary')
print(f'F1 Score on the test set: {f1}')
print(f'Accuracy of the model on the test set: {100 * correct / total}%')
