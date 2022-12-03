import board
import neopixel
import pigpio
import RPi.GPIO as GPIO
from gpiozero import LED
import time
# 将服务设置为开机启动
# sudo systemctl enable pigpiod
TrackingPin = 22 #感測器1
TrackingPin2 = 9 #感測器2
motor=LED(4) #馬達
led_4=LED(2)
pin_servo = 27

pwm = pigpio.pi()
pwm.set_mode(pin_servo, pigpio.OUTPUT)
pwm.set_PWM_frequency(pin_servo, 50)  # 50Hz frequency
def destroy():
    pwm.set_PWM_dutycycle(pin_servo, 0)
    pwm.set_PWM_frequency(pin_servo, 0)
# 輸入0 ～ 180度即可
# 別超過180度
def setDirection(angle):
    # 0 = 停止轉動
    # 500 = 0度
    # 1500 = 90度
    # 2500 = 180度
    duty = 500 + (angle / 180) * 2000
    pwm.set_servo_pulsewidth(pin_servo, duty)
    print("角度=", angle, "-> duty=", duty)
def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TrackingPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(TrackingPin2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def loop(pixels,TrackingPin,motor):
    x=0
    while True:
        if GPIO.input(TrackingPin) == GPIO.HIGH and x == 0:
            print ('目前偵測到')
            pixels.fill((0,255,0))
            pixels.show()
            motor.on()
            time.sleep(5)
            x=1
        elif GPIO.input(TrackingPin) == GPIO.HIGH and x == 1:
            motor.off()
            break
        else:    
            print ('沒有')
def loop2(TrackingPin2,led_4):
    while True:
        if GPIO.input(TrackingPin2) == GPIO.HIGH:
            print ('目前偵測到')
            led_4.on()
            time.sleep(7)
            led_4.off()
            setDirection(30)
            break
        else:    
            print ('沒有')
def destroy():
 GPIO.cleanup()
#-----------
#destroy()#清除針腳
length = 48
pixels = neopixel.NeoPixel(board.D10, length, brightness=0.1, auto_write=False,pixel_order=neopixel.RGB)
pixels.fill((255,0,0))
pixels.show()
setup()
loop(pixels,TrackingPin,motor)
loop2(TrackingPin2,led_4)
time.sleep(5)
setDirection(0)
pixels.fill((255,0,0))
pixels.show()
print("已完成")

#開啟綠燈
#偵測到後變紅燈,開起馬達結束
#偵測木板上的,偵測後開燈拍照
#伺服馬達送出紙張
#結束後燈光變綠



#https://learn.adafruit.com/circuitpython-on-raspberrypi-linux/installing-circuitpython-on-raspberry-pi
#pip3 install rpi_ws281x adafruit-circuitpython-neopixel
#python3 -m pip install --force-reinstall adafruit-blinka