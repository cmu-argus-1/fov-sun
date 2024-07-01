import matplotlib.pyplot as plt

from icosphere import icosphere # https://github.com/vedranaa/icosphere


nu = 10  # or any other integer
vertices, faces = icosphere(nu)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='b', s=1)
print(vertices)
plt.show()