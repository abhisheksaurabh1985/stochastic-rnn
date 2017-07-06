import time

def train(sess, loss_op, solver, nepochs, n_samples, learning_rate, batch_size, 
                    display_step, _X, data):
    avg_vae_loss = []
    start_time= time.time()
    print "###### Training starts ######"
    for epoch in range(nepochs):
        avg_cost= 0
        total_batch = int(n_samples/ batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = data.train.next_batch(batch_size)
            _, cost= sess.run([solver, loss_op], feed_dict={_X: batch_xs})
            avg_cost += (cost/ n_samples) *  batch_size
        avg_vae_loss.append(avg_cost)
        if epoch % display_step == 0:
            line =  "Epoch: %i \t Average cost: %0.9f" % (epoch, avg_vae_loss[epoch])
            print line
            # with open(logfile,'a') as f:
            #    f.write(line + "\n")
            # samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})
    print("--- %s seconds ---" % (time.time() - start_time))    
    return avg_vae_loss

