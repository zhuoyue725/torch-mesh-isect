SMPLX_JOINT_NAMES_55 = [ # 1+54 身体全部部位
    'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3', 'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist',
    'jaw','left_eye_smplhf','right_eye_smplhf','left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3','right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'
]

SMPL_JOINT_NAMES = [ # 1+23
    'Pelvis', 'L_Hip',  'R_Hip','Spine1',  'L_Knee',  'R_Knee', 'Spine2',  'L_Ankle', 'R_Ankle',  'Spine3',  'L_Foot',  'R_Foot',  
    'Neck',  'L_Collar',  'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',  'L_Elbow',  'R_Elbow', 'L_Wrist',  'R_Wrist', 'L_Hand', 'R_Hand'
]

SMPLX_BODY_JOINT_NAMES = [ # SMPLX的21个身体关节θ(去除第一个pelvis)
    # "pelvis",
    "left_hip",
    "right_hip", # 1
    "spine1",
    "left_knee",
    "right_knee", # 4
    "spine2",
    "left_ankle", # 6
    "right_ankle", # 7
    "spine3",
    "left_foot",
    "right_foot",
    "neck", # 11
    "left_collar",
    "right_collar", # 13
    "head",   # 14
    "left_shoulder",
    "right_shoulder", # 16
    "left_elbow",
    "right_elbow", # 18
    "left_wrist",
    "right_wrist", # 20
]

# 9上胸，6中胸，3肚子
# 13左肩，16左上臂，18 左小臂；14右肩，17右上臂，19 右小臂
# 12颈部，15头部，23左眼，24右眼
# 1左大腿，2右大腿