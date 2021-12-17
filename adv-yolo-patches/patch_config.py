from torch import optim


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.img_dir = "/home/rdr2143/waymo-adv-dataset/train-large-v2"
        self.lab_dir = "/home/rdr2143/waymo-adv-dataset/train-large-v2/labels"
        # self.img_dir = "inria/Train/pos"
        # self.lab_dir = "inria/Train/pos/yolo-labels"
        self.cfgfile = "cfg/yolo.cfg"
        self.weightfile = "weights/yolo.weights"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 300

        self.start_learning_rate = 0.03

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0

        self.batch_size = 20

        self.loss_target = lambda obj, cls: obj * cls


class Experiment1(BaseConfig):
    """
    Model that uses a maximum total variation, tv cannot go below this point.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment1'
        self.max_tv = 0.165


class Experiment2HighRes(Experiment1):
    """
    Higher res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 400
        self.patch_name = 'Exp2HighRes'

class Experiment3LowRes(Experiment1):
    """
    Lower res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 100
        self.patch_name = "Exp3LowRes"

class Experiment4ClassOnly(Experiment1):
    """
    Only minimise class score.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment4ClassOnly'
        self.loss_target = lambda obj, cls: cls




class Experiment1Desktop(Experiment1):
    """
    """

    def __init__(self):
        """
        Change batch size.
        """
        super().__init__()

        self.batch_size = 8
        self.patch_size = 400


class ReproducePaperObj(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 8
        self.patch_size = 300

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

class WaymoApplierObj(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """
# DIR_TO_STORE = '/home/rdr2143/waymo-adv-dataset/adv/'
# DIR_TO_STORE = '/home/rdr2143/inria-adv-dataset/'
# DIR_TO_STORE = '/home/rdr2143/waymo-adv-dataset/trained-patches/'

    def __init__(self):
        super().__init__()
        # self.img_dir = "inria/Train/pos"
        # self.lab_dir = "inria/Train/pos/yolo-labels"
        # self.dir_to_store = '/home/rdr2143/inria-adv-dataset/'

        # self.img_dir = "/home/rdr2143/waymo-adv-dataset/train"
        # self.lab_dir = "/home/rdr2143/waymo-adv-dataset/train/labels"
        # self.dir_to_store = '/home/rdr2143/waymo-adv-dataset/adv/'

        # self.img_dir = "/home/rdr2143/inria-adv-dataset/single-failed-v2"
        # self.lab_dir = "/home/rdr2143/inria-adv-dataset/single-failed-v2/labels"
        # self.dir_to_store = "/home/rdr2143/inria-adv-dataset/single-failed-v2-pose/"
        
        # self.img_dir = "/home/rdr2143/inria-adv-dataset/single-failed-v2"
        # self.lab_dir = "/home/rdr2143/inria-adv-dataset/single-failed-v2/yolo-labels"
        # self.dir_to_store = "/home/rdr2143/inria-adv-dataset/single-failed-v2-regular/"

        # self.img_dir = "/home/rdr2143/waymo-adv-dataset/single-failed-v1"
        # self.lab_dir = "/home/rdr2143/waymo-adv-dataset/single-failed-v1/yolo-labels"
        # self.dir_to_store = "/home/rdr2143/waymo-adv-dataset/single-failed-v1-regular/"

        self.img_dir = "/home/rdr2143/waymo-adv-dataset/single-failed-v1"
        self.lab_dir = "/home/rdr2143/waymo-adv-dataset/single-failed-v1/labels"
        self.dir_to_store = "/home/rdr2143/waymo-adv-dataset/single-failed-v1-pose/"

        self.batch_size = 1
        self.patch_size = 300

        self.patch_name = 'WaymoApplierObj'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

patch_configs = {
    "base": BaseConfig,
    "exp1": Experiment1,
    "exp1_des": Experiment1Desktop,
    "exp2_high_res": Experiment2HighRes,
    "exp3_low_res": Experiment3LowRes,
    "exp4_class_only": Experiment4ClassOnly,
    "paper_obj": ReproducePaperObj,
    "waymo_applier_obj": WaymoApplierObj
}
