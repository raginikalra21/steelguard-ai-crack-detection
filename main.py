from src.preprocessing import get_generators

train_gen, val_gen = get_generators()

print(train_gen.class_indices)
print("Train samples:", train_gen.samples)
print("Validation samples:", val_gen.samples)

from src.utils import get_class_weights

class_weights = get_class_weights(train_gen)
print(class_weights)