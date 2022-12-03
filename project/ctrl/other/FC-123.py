import RPi.GPIO as GPIO
TrackingPin = 9

def setup():
 GPIO.setmode(GPIO.BCM)
 GPIO.setup(TrackingPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def loop():
 while True:
  if GPIO.input(TrackingPin) == GPIO.LOW:
   # Detectie wit oppervalk (of voorwerp bij sensor)
   print ('[wit | object]')

  else:
   # detectie zwart oppervlak (of geen voorwerp bij sensor)
   print ('[zwart | geen object]')

def destroy():
 GPIO.cleanup()

destroy()
setup()
loop()


destroy()