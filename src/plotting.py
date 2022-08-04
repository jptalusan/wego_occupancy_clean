import matplotlib.pyplot as plt

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss", marker='o')
    plt.plot(epochs, val_loss, "r", label="Validation loss", marker='s')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    return plt

def create_time_steps(length):
    return list(range(-length, 0))
    
# def multi_step_plot(history, true_future, prediction):
#     plt.figure(figsize=(18, 6))
#     num_in = create_time_steps(len(history))
#     num_out = len(true_future)

#     plt.plot(num_in, np.array(history[:, -1]), label='History', marker='^')
#     plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
#            label='True Future', markersize=10)
#     plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'r*',
#            label='Predicted Future')
#     plt.legend(loc='upper left')
#     plt.show()
    
def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors)+1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)     
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values
    
    dcolorscale = [] #discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale