from main_net import Main_net
import torch
from loss_f import LapLoss
from atd12k import get_loader
from softcode import myutils

data_dir = '/home/jiaming/atd12k_points'

train_loader = get_loader('train', data_dir, 2, shuffle=True, num_workers=1)
test_loader = get_loader('test', data_dir, 2, shuffle=False, num_workers=1)

# class LaplacianPyramid(nn.Module):
#     def __init__(self, max_level=5):
#         super(LaplacianPyramid, self).__init__()
#         self.gaussian_conv = GaussianConv()
#         self.max_level = max_level
#
#     def forward(self, X):
#         t_pyr = []
#         current = X
#         for level in range(self.max_level):
#             t_guass = self.gaussian_conv(current)
#             t_diff = current - t_guass
#             t_pyr.append(t_diff)
#             current = F.avg_pool2d(t_guass, 2)
#         t_pyr.append(current)
#
#         return t_pyr
#
# class LaplacianLoss(nn.Module):
#     def __init__(self):
#         super(LaplacianLoss, self).__init__()
#
#         self.criterion = nn.L1Loss()
#         self.lap = LaplacianPyramid()
#
#     def forward(self, x, y):
#         x_lap, y_lap = self.lap(x), self.lap(y)
#         return sum(self.criterion(a, b) for a, b in zip(x_lap, y_lap))

def train():
    batch_size = 6
    total_step = 100
    num_workers = 0

    lr = 1e-4
    criteration = LapLoss()
    H = 192
    W = 384
    shape= [batch_size,3,H,W]
    model = Main_net(shape).cuda().train(True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=4e-4)

    for step in range(total_step):
        total_loss = 0
        psnrs, ssims = myutils.init_meters()

        for ix,data in enumerate(train_loader):

            img1,img2 ,tar= data
            img1 = img1.cuda()
            img2 = img2.cuda()
            tar  = tar.cuda()
            img_out  = model(img1,img2)
            # loss = torch.nn.functional.l1_loss(img_out,tar)
            loss = criteration(img_out, tar)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            if ix %500==0:
                print('data idx:' +' lr  :'+str(lr)+'  epoch:  ' +str(ix)+'  /  '+str(len(train_loader)))
                print('loss value :', loss.item())
            total_loss+=loss

        print('epoch:  '+str(step)+'    avg loss   :'+str(total_loss.item()/len(train_loader)))
        if (step+1)%3==0:
            test(step)
            torch.save(model,'./weights/'+'model_weight_'+str(step+1)+'.pth')


def test(step):
    batch_size = 4
    total_step = 100
    num_workers = 0
    H = 192
    W = 384
    lr = 1e-4
    criteration = LapLoss()
    shape = [batch_size, 3, H, W]
    model = Main_net(shape).cuda().train(True)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=4e-4)
    model.eval()
    criteration.eval()
    psnrs, ssims = myutils.init_meters()

    with torch.no_grad():
        for ix, data in enumerate(test_loader):

            img1, img2, tar, dir = data
            img1 = img1.cuda()
            img2 = img2.cuda()
            tar = tar.cuda()
            img_out = model(img1, img2)
            # loss = torch.nn.functional.l1_loss(img_out,tar)
            loss = criteration(img_out, tar)
            myutils.eval_metrics(img_out, tar, psnrs, ssims)

            if ix % 500 == 0:
                print('data idx:' + ' lr  :' + str(lr) + '  epoch:  ' + str(ix) + '  /  ' + str(len(train_loader)))
                print('loss value :', loss.item())
            # total_loss += loss

        print('epoch:  ' + str(step) + '    psnr :' + str(psnrs.avg) + '    ssim :' + str(ssims.avg) )


if __name__=='__main__':
    train()