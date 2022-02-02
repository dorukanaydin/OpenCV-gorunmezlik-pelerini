import cv2
import numpy as np

#Görünmezlik pelerini denemesi

cam = cv2.VideoCapture(0)

#Buradan pelerin olarak kullanılacak cismin renk ayarları yapılabilir.Burada kırmızı rengin bir tonu kullanıldı.
lower = np.array([0,169,96])
upper = np.array([159,255,195])

#Kamera ilk açıldığında arkaplan görüntüsü aldı.Burada bizim bulunmadığımız arkaplan görüntüsü kullanılmalıdır.
_,background = cam.read()

kernel = np.ones((3,3),np.uint8)
kernel2 = np.ones((11,11),np.uint8)
kernel3 = np.ones((13,13),np.uint8)


while cam.isOpened():
    _, frame = cam.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #mask = sadece kırmızı kısımlar görünür.(etraf siyah)
    mask = cv2.inRange(hsv, lower, upper)
    """
    Önce mask'deki gürültü temizlendi.(close)
    Sonra morph_open kullanıldı.
    En son görüntü biraz daha büyütüldü ve temiz bir görüntü elde edildi.(dilate)
    
    """
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel2)
    mask = cv2.dilate(mask,kernel3,iterations=2)
    
    mask_not = cv2.bitwise_not(mask)
    
    #bg'de arka plan resmi ile mask çarpıldı.Böylece sadece kırmızı kısımlar göründü.
    bg = cv2.bitwise_and(background,background,mask=mask)
    #fg'de arka plan renkli kırmızılar siyah göründü.
    fg = cv2.bitwise_and(frame,frame,mask=mask_not)
    
    dst = cv2.addWeighted(bg,1,fg,1,0)
    
    cv2.imshow("Orjinal",frame)
    #cv2.imshow("Mask",mask)
    cv2.imshow("Dst",dst)
    
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()  
    
    