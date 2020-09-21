from string import Template

from federatedml.nn.backend.tf_keras.nn_model import _modify_model_input_shape


def build_model():
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Dense, Dropout, Flatten, Reshape
    from keras.models import Sequential

    model = Sequential()
    model.add(Reshape((28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


def save_runtime_conf(m):
    nn_define = m.to_json()
    temp = open("mnist_conf_temperate.json").read()
    json = Template(temp).substitute(nn_define=nn_define)

    with open("mnist_conf.json", "w") as f:
        f.write(json)


if __name__ == '__main__':
    m = build_model()
    save_runtime_conf(m)
