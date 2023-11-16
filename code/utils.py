def save_interaction_scores(model, loader, output_filename):
    model.eval()
    interaction_scores = []
    with torch.no_grad():
        for inputs, _ in loader:
            class_idx = model(inputs).argmax(dim=1).unsqueeze(dim=1)
            interaction = get_second_order_grad(model, inputs, class_idx)

            for i in range(interaction.shape[0]):
                for j in range(interaction.shape[1]):
                    interaction_scores.append([i, j, interaction[i, j].item()])

    # Save to file
    np.savetxt(output_filename, interaction_scores, delimiter=",", header="Feature1,Feature2,InteractionScore")



# Save interaction scores for test data
# Usage: save_interaction_scores(mlp_model, test_loader, "interaction_scores.csv")


def get_second_order_grad(model, x, class_idx, num_genes):
    with torch.set_grad_enabled(True):

        if x.nelement() < 2:
            return np.array([])

        x.requires_grad = True
        output, x1 = model.forward_plus(x)
        y = torch.gather(output, 1, class_idx).sum(dim=0)

        grads = autograd.grad(y, x1, create_graph=True, allow_unused=True)[0].sum(dim=0)  # [valid_index]

        grad_list = []
        for j, grad in enumerate(grads):
            grad2 = autograd.grad(grad, x1, retain_graph=True)[0]  # .sum(dim=0)  # [valid_index]
            grad_list.append(grad2.detach())

    with torch.no_grad():

        grad_matrix = torch.stack(grad_list).permute((1, 0, 2))
        del grad_list
        grad_matrix0 = torch.zeros((num_genes, num_genes), device=x.device)
        weight1 = model.layer1_weight()

        for idx in range(grad_matrix.shape[0]):
            grad_matrix0 += weight1.T.mm(grad_matrix[idx]).mm(weight1).abs()

    return grad_matrix0

def save_checkpoint(fname, **kwargs):

    checkpoint = {}

    for key, value in kwargs.items():
        checkpoint[key] = value
        # setattr(self, key, value)

    torch.save(checkpoint, fname)

