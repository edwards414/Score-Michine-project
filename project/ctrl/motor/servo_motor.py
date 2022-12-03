#coding:utf-8
import pigpio
import RPi.GPIO as GPIO
import time
pin_servo = 27
# 終端機需輸入(才能執行pigpiod)
# sudo systemctl enable pigpiod
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
setDirection(100)
time.sleep(5)
setDirection(130)
#time.sleep(5)
#setDirection(100)
