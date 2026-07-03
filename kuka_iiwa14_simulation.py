import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("/home/host-20-04/mujoco_workspace/kuka_iiwa14_RL/mjcf/kuka_iiwa14_scene.xml")
data = mujoco.MjData(model)

reverse = False
def motion_control(data):
    global reverse
    if data.ctrl[0] < 2.96706 and not reverse:
        data.ctrl[0] += 0.001
    elif data.ctrl[0] > -2.96706:
        reverse = True
        data.ctrl[0] -= 0.001
    else:
        reverse = False

def randomize_trg_pos(model, data):
    new_x_pos =  np.random.uniform(0.5, 0.8)
    new_y_pos =  np.random.uniform(0.5, 0.8)
    new_z_pos =  np.random.uniform(0.1, 1.0)

    model.site_pos[0][0] = new_x_pos
    model.site_pos[0][1] = new_y_pos
    model.site_pos[0][2] = new_z_pos

    data.site_xpos[0][0] = new_x_pos
    data.site_xpos[0][1] = new_y_pos
    data.site_xpos[0][2] = new_z_pos

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        
        motion_control(data)

        randomize_trg_pos(model, data)

        # print(data.qpos[0], data.ctrl[0])
        # print(len(data.qacc))
        print(data.site_xpos[0])
        # print(mujoco.mj_name2id(
        #       model,
        #       mujoco.mjtObj.mjOBJ_SITE,
        #       "target_site",
        #       )
        #     )

        mujoco.mj_step(model, data)
        
        viewer.sync()
