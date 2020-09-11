from multiprocessing import Process
from multiprocessing.managers import BaseManager
import time

from files.world_listener import WorldListener
from files.eye_listener import EyeListener
from files.object.object_detect import ObjectDetect
from files.do_stuff import DoStuff
from files.do_stuff_with_combined_eye import DoStuffWithCombinedEye
from files.do_stuff_together import DoStuffTogether
from files.logger import logger

from files.world import World
from files.pupil import Pupil
from files.common_data import CommonData


def start_process(glass_id, glass_port, common_data_proxy, world_proxy, eye_0_proxy, eye_1_proxy, object_detect_proxy):
    try:
        world = WorldListener(glass_id, glass_port)
        eye_0 = EyeListener(glass_id, 0, glass_port)
        eye_1 = EyeListener(glass_id, 1, glass_port)
        #print(world.world_receiver)
        #print(eye_0.eye_receiver)
        #print(eye_1.eye_receiver)

        do_stuff = DoStuff(glass_id, confidence_threshold, num_objects, object_detect_proxy, debug)

        world_receiver = Process(target=world.world_receiver, args=(world_proxy,),
                                 name='frame_world_glass_{}'.format(glass_id))
        eye_0_receiver = Process(target=eye_0.eye_receiver, args=(eye_0_proxy,),
                                 name='gaze_0_glass_{}'.format(glass_id))
        eye_1_receiver = Process(target=eye_1.eye_receiver, args=(eye_1_proxy,),
                                 name='gaze_1_glass_{}'.format(glass_id))
        do_some_stuff = Process(target=do_stuff.do_some_stuff,
                                args=(world_proxy, eye_0_proxy, eye_1_proxy, common_data_proxy),
                                name='do_stuff_glass_{}'.format(glass_id))

    except Exception as e:
        raise e

    return world_receiver, eye_0_receiver, eye_1_receiver, do_some_stuff


def start_process_with_combined_eye(glass_id, glass_port, common_data_proxy, world_proxy, eye_0_proxy, object_detect_proxy):
    try:
        world = WorldListener(glass_id, glass_port)
        eye_0 = EyeListener(glass_id, '', glass_port)

        do_stuff = DoStuffWithCombinedEye(glass_id, confidence_threshold, num_objects, object_detect_proxy, debug)

        world_receiver = Process(target=world.world_receiver, args=(world_proxy,),
                                 name='frame_world_glass_{}'.format(glass_id))
        eye_0_receiver = Process(target=eye_0.eye_receiver, args=(eye_0_proxy,),
                                 name='gaze_0_glass_{}'.format(glass_id))
        do_some_stuff = Process(target=do_stuff.do_some_stuff, args=(world_proxy, eye_0_proxy, common_data_proxy),
                                name='do_stuff_glass_{}'.format(glass_id))

    except Exception as e:
        raise e

    return world_receiver, eye_0_receiver, do_some_stuff


def main():

    # Start glass 1 processes
    world_receiver_1, eye_0_receiver_1, eye_1_receiver_1, do_some_stuff_1 = start_process(1, port_glass_1,
                                                                                          common_data_proxy_1,
                                                                                          world_proxy_glass_1,
                                                                                          eye_0_proxy_glass_1,
                                                                                          eye_1_proxy_glass_1,
                                                                                          object_detect_proxy_glass_1)
    world_receiver_1.start()
    eye_0_receiver_1.start()
    eye_1_receiver_1.start()
    do_some_stuff_1.start()

    # Start glass 2 processes
    world_receiver_2, eye_0_receiver_2, eye_1_receiver_2, do_some_stuff_2 = start_process(2, port_glass_2,
                                                                                          common_data_proxy_2,
                                                                                          world_proxy_glass_2,
                                                                                          eye_0_proxy_glass_2,
                                                                                          eye_1_proxy_glass_2,
                                                                                          object_detect_proxy_glass_2)
    world_receiver_2.start()
    eye_0_receiver_2.start()
    eye_1_receiver_2.start()
    do_some_stuff_2.start()

    # Start final combined process
    stuff_together = Process(target=do_stuff_together.do_some_stuff_together,
                             args=(common_data_proxy_1, common_data_proxy_2), name="do_stuff_together")
    stuff_together.start()

    time.sleep(1)
    logger.info("Application started successfully")

    world_receiver_1.join()
    eye_0_receiver_1.join()
    eye_1_receiver_1.join()
    do_some_stuff_1.join()

    world_receiver_2.join()
    eye_0_receiver_2.join()
    eye_1_receiver_2.join()
    do_some_stuff_2.join()

    stuff_together.join()


def main_with_combined_eye():

    # Start glass 1 processes
    world_receiver_1, eye_0_receiver_1, do_some_stuff_1 = start_process_with_combined_eye(1, port_glass_1,
                                                                                          common_data_proxy_1,
                                                                                          world_proxy_glass_1,
                                                                                          eye_0_proxy_glass_1,
                                                                                          object_detect_proxy_glass_1)
    world_receiver_1.start()
    eye_0_receiver_1.start()
    do_some_stuff_1.start()
    #print(port_glass_1)
    # Start glass 2 processes
    world_receiver_2, eye_0_receiver_2, do_some_stuff_2 = start_process_with_combined_eye(2, port_glass_2,
                                                                                          common_data_proxy_2,
                                                                                          world_proxy_glass_2,
                                                                                          eye_0_proxy_glass_2,
                                                                                          object_detect_proxy_glass_2)
    world_receiver_2.start()
    eye_0_receiver_2.start()
    do_some_stuff_2.start()
    
    # Start final combined process
    stuff_together = Process(target=do_stuff_together.do_some_stuff_together,
                             args=(common_data_proxy_1, common_data_proxy_2), name="do_stuff_together")
    stuff_together.start()

    time.sleep(1)
    logger.info("Application started successfully")

    world_receiver_1.join()
    eye_0_receiver_1.join()
    do_some_stuff_1.join()

    world_receiver_2.join()
    eye_0_receiver_2.join()
    do_some_stuff_2.join()

    stuff_together.join()


if __name__ == "__main__":
    logger.info("Starting application...")

    # input parameters to the application
    port_glass_1 = 50020
    port_glass_2 = 50021
    confidence_threshold = 0.30
    num_objects = 6
    use_both_eyes = True
    debug = True

    # Proxy objects for common data for both glasses
    BaseManager.register('CommonData_1', CommonData)
    manager_1 = BaseManager()
    manager_1.start()
    common_data_proxy_1 = manager_1.CommonData_1()

    BaseManager.register('CommonData_2', CommonData)
    manager_2 = BaseManager()
    manager_2.start()
    common_data_proxy_2 = manager_2.CommonData_2()

    # Proxy objects for world and pupil for Glass 1
    BaseManager.register('World_Glass1', World)
    manager_world_glass_1 = BaseManager()
    manager_world_glass_1.start()
    world_proxy_glass_1 = manager_world_glass_1.World_Glass1()

    BaseManager.register('Pupil_0_Glass1', Pupil)
    manager_eye_0_glass_1 = BaseManager()
    manager_eye_0_glass_1.start()
    eye_0_proxy_glass_1 = manager_eye_0_glass_1.Pupil_0_Glass1()

    BaseManager.register('Pupil_1_Glass1', Pupil)
    manager_eye_1_glass_1 = BaseManager()
    manager_eye_1_glass_1.start()
    eye_1_proxy_glass_1 = manager_eye_1_glass_1.Pupil_1_Glass1()

    # Proxy objects for world and pupil for Glass 2
    BaseManager.register('World_Glass2', World)
    manager_world_glass_2 = BaseManager()
    manager_world_glass_2.start()
    world_proxy_glass_2 = manager_world_glass_2.World_Glass2()

    BaseManager.register('Pupil_0_Glass2', Pupil)
    manager_eye_0_glass_2 = BaseManager()
    manager_eye_0_glass_2.start()
    eye_0_proxy_glass_2 = manager_eye_0_glass_2.Pupil_0_Glass2()

    BaseManager.register('Pupil_1_Glass2', Pupil)
    manager_eye_1_glass_2 = BaseManager()
    manager_eye_1_glass_2.start()
    eye_1_proxy_glass_2 = manager_eye_1_glass_2.Pupil_1_Glass2()

    # Object Detection proxy objects
    BaseManager.register('ObjectDetect_Glass_1', ObjectDetect)
    manager_object_detect_glass_1 = BaseManager()
    manager_object_detect_glass_1.start()
    object_detect_proxy_glass_1 = manager_object_detect_glass_1.ObjectDetect_Glass_1()

    BaseManager.register('ObjectDetect_Glass_2', ObjectDetect)
    manager_object_detect_glass_2 = BaseManager()
    manager_object_detect_glass_2.start()
    object_detect_proxy_glass_2 = manager_object_detect_glass_2.ObjectDetect_Glass_2()

    do_stuff_together = DoStuffTogether()

    if use_both_eyes:
        main()
    else:
        main_with_combined_eye()
