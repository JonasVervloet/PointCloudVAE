import numpy as np
import matplotlib.pyplot as plt

PATH = "C:/Users/vervl/OneDrive/Documenten/Studie/2019-2020/Masterproef/Images/"

path1 = "D:/Resultaten/AirplaneVAE8/"
epoch_nb1 = 225
alfa1 = 1.0
alfa2 = 1.0

path2 = "D:/Resultaten/AirplaneVAE9/"
epoch_nb2 = 225
alfa3 = 1.0
alfa4 = 1.0

path3 = "D:/Resultaten/AirplaneVAE11/"
epoch_nb3 = 225
alfa5 = 1.0
alfa6 = 1.0

# path4 = "D:/Resultaten/AirplaneVAE4/"
# epoch_nb4 = 225
# alfa7 = 2.0
# alfa8 = 2.0
#
# path5 = "D:/Resultaten/AirplaneVAE8/"
# epoch_nb5 = 225
# alfa9 = 1.0
# alfa10 = 1.0


losses1_train = np.load(path1 + "trainloss_epoch{}.npy".format(epoch_nb1))
losses1_val = np.load(path1 + "valloss_epoch{}.npy".format(epoch_nb1))

losses2_train = np.load(path2 + "trainloss_epoch{}.npy".format(epoch_nb2))
losses2_val = np.load(path2 + "valloss_epoch{}.npy".format(epoch_nb2))

losses3_train = np.load(path3 + "trainloss_epoch{}.npy".format(epoch_nb3))
losses3_val = np.load(path3 + "valloss_epoch{}.npy".format(epoch_nb3))

# losses4_train = np.load(path4 + "trainloss_epoch{}.npy".format(epoch_nb4))
# losses4_val = np.load(path4 + "valloss_epoch{}.npy".format(epoch_nb4))
#
# losses5_train = np.load(path5 + "trainloss_epoch{}.npy".format(epoch_nb5))
# losses5_val = np.load(path5 + "valloss_epoch{}.npy".format(epoch_nb5))


losses1_train *= alfa1
losses1_val *= alfa2

losses2_train *= alfa3
losses2_val *= alfa4

losses3_train *= alfa5
losses3_val *= alfa6

# losses4_train *= alfa7
# losses4_val *= alfa8
#
# losses5_train *= alfa9
# losses5_val *= alfa10


print(losses1_train)

plt.clf()
x = range(epoch_nb1 + 1)
plt.plot(x, losses1_val, x, losses2_val, x, losses3_val)
plt.legend(['n=2', 'n=3', 'n=10'])
plt.title('gelaagde-VAE PrimitieveGeometrie (n=2)')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(
    PATH + "latent_spaces.png"
)
plt.show()

# plt.clf()
# x = range(epoch_nb2 + 1)
# plt.plot(x, losses2_train, x, losses2_val)
# plt.legend(['train loss', 'validatie loss'])
# plt.title('gelaagde-VAE ShapeNet (n=2)')
# plt.yscale('log')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.savefig(
#     path2 + "loss_epoch{}.png".format(epoch_nb2)
# )
# plt.show()
