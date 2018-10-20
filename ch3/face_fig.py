from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

ppl = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = ppl.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={
                         'xticks': (), 'yticks': ()})

for target, image, ax in zip(ppl.target, ppl.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(ppl.target_names[target])

fig.savefig("faces.png")
print(f"people.images.shape: {image_shape}")
print(f"Number of classes: {len(ppl.target_names)}")
