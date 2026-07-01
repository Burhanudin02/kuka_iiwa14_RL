import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("/home/host-20-04/mujoco_workspace/shared/models/kuka_iiwa14_scene.xml")
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

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        
        motion_control(data)

        # print(data.qpos[0], data.ctrl[0])
        # print(len(data.qacc))
        # print(data.site_xpos[0])
        print(mujoco.mj_name2id(
              model,
              mujoco.mjtObj.mjOBJ_SITE,
              "target_site",
              )
            )

        mujoco.mj_step(model, data)
        
        viewer.sync()
