
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    #elif opt.model == 'pix2pix':
    #    assert(opt.dataset_mode == 'aligned')
    #    from .pix2pix_model import Pix2PixModel
    #    model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'pix2pix3d':
        assert(opt.dataset_mode == 'nodule')
        from .pix2pix3d_model import Pix2Pix3dModel
        model = Pix2Pix3dModel()
    elif opt.model == 'cycle_gan3d':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan3d_model import CycleGAN3dModel
        model = CycleGAN3dModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
