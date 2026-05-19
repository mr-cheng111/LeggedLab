from pxr import Usd, Gf

usd_path = "/home/mr-cheng/Project/LeggedLab/legged_lab/assets/xuanji/rb160w/usd/rb160w.usd"
stage = Usd.Stage.Open(usd_path)

j = stage.GetPrimAtPath("/RB160W/joints/RR_calf_joint")
q = Gf.Quatf(1.0, 0.0, 0.0, 0.0)

j.GetAttribute("physics:localRot0").Set(q)
j.GetAttribute("physics:localRot1").Set(q)

stage.GetRootLayer().Save()
