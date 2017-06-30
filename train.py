def train(dataset, n_samples, learning_rate=0.001, batch_size=100, 
          training_epochs=10, display_step=5):
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = dataset.train.next_batch(batch_size) # Rename _ to batch_xs_labels. batch_xs: (100, 784); _: (100, 10)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), 
                  "cost=", "{:.9f}".format(avg_cost))
    return vae