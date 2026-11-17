# Generates the synthetic placeholder sketches used only to test the pipeline.
# Synthetic "pencil" sketches on a noisy, unevenly lit paper background, to test preprocessing.
import cv2, numpy as np
rng=np.random.default_rng(0)
def paper(h=768,w=640):
    g=np.full((h,w),235,np.float32)
    g+=np.linspace(-25,10,w)[None,:]  # uneven light
    g+=rng.normal(0,4,(h,w)); return g
def save(name,g):
    g=np.clip(g,0,255).astype(np.uint8); import os; os.makedirs("test_sketches",exist_ok=True); cv2.imwrite(f"test_sketches/{name}",g)
def L(g,p,q,t=3): cv2.line(g,p,q,70,t,cv2.LINE_AA)
g=paper()
pts=np.array([[230,700],[250,200],[320,80],[390,200],[410,700]]);cv2.polylines(g,[pts],False,70,3)
L(g,(150,700),(500,700)); 
for y in range(240,690,45): L(g,(255,y),(395,y),2)
L(g,(320,80),(320,20),2); cv2.ellipse(g,(320,330),(110,25),0,0,360,70,2)
save("building.png",g)
g=paper()
cv2.rectangle(g,(220,120),(420,280),70,3); cv2.circle(g,(280,190),22,70,3); cv2.circle(g,(360,190),22,70,3)
L(g,(285,245),(355,245)); L(g,(320,120),(320,70)); cv2.circle(g,(320,60),12,70,3)
cv2.rectangle(g,(200,300),(440,560),70,3); cv2.rectangle(g,(270,360),(370,460),70,2)
L(g,(200,330),(130,470)); L(g,(440,330),(510,470)); L(g,(260,560),(250,700)); L(g,(380,560),(390,700))
L(g,(320,280),(320,300))
save("character.png",g)
g=paper()
L(g,(200,420),(460,420)); L(g,(200,420),(230,480)); L(g,(460,420),(430,480)); L(g,(230,480),(430,480))
L(g,(220,480),(200,700)); L(g,(440,480),(460,700)); L(g,(260,480),(270,650)); L(g,(400,480),(390,650))
cv2.ellipse(g,(330,300),(130,120),0,180,360,70,3); L(g,(200,300),(200,420)); L(g,(460,300),(460,420))
save("product.png",g)
