from opts import opts
from detectors.detector_factory import detector_factory

opt = opts().init([])
opt.task = 'multi_pose'
opt.dataset = 'carfusion'
opt.arch = 'res_50'
opt.load_model = '../exp/multi_pose/carfusion_pose_r50/model_last.pth'
opt.demo = '../data/carfusion/images/val/car_morewood1_10_12546.jpg'
opt.input_res = 512
opt.gpus = 0
opt.debug = 0
opt.exp_id = 'carfusion_pose_r50'

detector = detector_factory[opt.task](opt)
detector.run(opt.demo)