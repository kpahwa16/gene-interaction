def train_model(train_loader, test_loader, num_genes, learning_rate=0.001, epochs=10):
    # Check for CUDA availability
    device = "cpu"
    print(f"Using device: {device}")

    # Instantiate the model
    model = MLP(num_genes=num_genes,hidden1=2048, hidden2=512).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model
    model_save_path = './mlp_model.pth'
    torch.save(model.state_dict(), model_save_path)
