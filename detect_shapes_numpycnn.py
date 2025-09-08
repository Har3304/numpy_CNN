import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from train_numpy_cnn import NumPyCNN, LABEL_MAP

#Folder directories
dict = {
    'tri': 0,
    'circ': 1,
    'squ': 2,
    'rect': 3
}

data = []
labels = []

count=0
for folder, label in dict.items():
    path=r"F:\\sm"
    path += '\\'+folder 
    for file in os.listdir(path):
        if file.endswith(".png"):
            count+=1
            print(f'{count}/2500')
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
            img = cv2.resize(img, (64, 64)) 
            data.append(img.flatten())
            labels.append(label)

X = np.array(data)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SVC(kernel='rbf', C=10, gamma=0.01)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("press 0 for triangle")
print("press 1 for circle")
print("press 2 for square")
print("press 3 for rectangle")
print("press 4 to random selection")

choice=int(input("choose what shape you want to test: "))
if choice==4:
    choice=np.random.choice([0, 1, 2, 3])    
if choice==0:
    fig, ax = plt.subplots(figsize=(2, 2))    
    side = np.random.randint(30, 60)  
    angle = np.pi / 3  
    cx, cy = np.random.randint(30, 70), np.random.randint(30, 70)  
    x0 = cx
    y0 = cy+side/np.sqrt(3)
    x1 = cx-side/2
    y1 = cy-side/ (2 * np.sqrt(3))
    x2 = cx + side/2
    y2 = y1
    x_tri = [x0, x1, x2, x0]
    y_tri = [y0, y1, y2, y0]
    ax.fill(x_tri, y_tri, color=np.random.rand(3,), alpha=0.6)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(f"F:/sm/test_image/plot_test.png")
    plt.show()
    plt.close(fig)

elif choice==1:
    fig, ax = plt.subplots()
    r = np.random.randint(100) 
    x0, y0 = np.random.randint(100), np.random.randint(100)
    theta = np.linspace(0, 2 * np.pi, 500)
    x = r * np.cos(theta) + x0
    y = r * np.sin(theta) + y0
    ax.fill(x, y)
    ax.set_aspect('equal')
    ax.axis('off')    
    fig.savefig(f"F:/sm/test_image/plot_test.png")
    plt.show()
    plt.close(fig)

elif choice==2:
    fig, ax = plt.subplots()
    s = np.random.randint(10, 50)
    x0 = np.random.randint(100)
    y0 = np.random.randint(100)
    x_sq = [x0, x0 + s, x0 + s, x0]
    y_sq = [y0, y0, y0 + s, y0 + s]
    ax.fill(x_sq, y_sq)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(f"F:/sm/test_image/plot_test.png")
    plt.show()
    plt.close(fig)

else:
    fig, ax = plt.subplots()
    w, h = np.random.randint(20, 80), np.random.randint(10, 50)
    x0 = np.random.randint(100)
    y0 = np.random.randint(100)
    x_rect = [x0, x0 + w, x0 + w, x0]
    y_rect = [y0, y0, y0 + h, y0 + h]
    ax.fill(x_rect, y_rect)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(f"F:/sm/test_image/plot_test.png")
    plt.show()
    plt.close(fig)


test = []
path = 'F:/sm/test_image'
img_path = os.path.join(path, 'plot_test.png')
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
img = cv2.resize(img, (64, 64))  
img_norm = img.astype(np.float32) / 255.0
inp = img_norm[None, None, :, :]   
model = NumPyCNN()
model.load("cnn_weights.pkl")
y_pred_cnn = model.predict(inp)[0]


for name, label in dict.items():    
    if label == y_pred_cnn:
        print(f"NumPyCNN prediction {name}")

 
