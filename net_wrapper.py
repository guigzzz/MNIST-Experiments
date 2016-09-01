import network


def start_net(net_type):

    if net_type == 'test':

        training_data = [0.05, 0.1]
        checking_data = [0.01, 0.99]
        layers = [2, 2, 2]

    elif net_type == 'gate':
        training_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
        checking_data = [0, 0, 0, 1]
        layers = [2, 1, 1]

    epochs = 10000

    learning_rate = 10

    # initialise network

    net = network.network(training_data, checking_data, layers, epochs,
                          learning_rate, net_type)

    # run network

    net.run()


# net_type = 'test'
net_type = 'gate'
start_net(net_type)
