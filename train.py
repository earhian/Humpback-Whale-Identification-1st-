import datetime
import os

from timeit import default_timer as timer
from dataSet import *
from models import *
import torch
import time
from utils import *
from torch.nn.parallel.data_parallel import data_parallel



def train_collate(batch):

    batch_size = len(batch)
    images = []
    labels = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            labels.extend(batch[b][1])
    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))
    return images, labels

def valid_collate(batch):

    batch_size = len(batch)
    images = []
    labels = []
    names = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            labels.append(batch[b][1])
            names.append(batch[b][2])
    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))
    return images, labels, names
def transform_train(image, mask, label):
    add_ = 0
    image = cv2.resize(image, (512, 256))
    mask = cv2.resize(mask, (512, 256))
    mask = mask[:,:, None]

    image = np.concatenate([image, mask], 2)
    # if 0:
    #     if random.random() < 0.5:
    #         image = bgr_to_gray(image)

    if 1:
        if random.random() < 0.5:
            image = np.fliplr(image)
            if not label == 'new_whale':
                add_ += 5004
        image, mask = image[:,:,:3], image[:,:, 3]
    if random.random() < 0.5:
        image, mask = random_angle_rotate(image, mask, angles=(-25, 25))
    # noise
    if random.random() < 0.5:
        index = random.randint(0, 1)
        if index == 0:
            image = do_speckle_noise(image, sigma=0.1)
        elif index == 1:
            image = do_gaussian_noise(image, sigma=0.1)
    if random.random() < 0.5:
        index = random.randint(0, 3)
        if index == 0:
            image = do_brightness_shift(image,0.1)
        elif index == 1:
            image = do_gamma(image, 1)
        elif index == 2:
            image = do_clahe(image)
        elif index == 3:
            image = do_brightness_multiply(image)
    if 1:
        image, mask = random_erase(image,mask, p=0.5)
    if 1:
        image, mask = random_shift(image,mask, p=0.5)
    if 1:
        image, mask = random_scale(image,mask, p=0.5)
    # todo data augment
    if 1:
        if random.random() < 0.5:
            mask[...] = 0
    mask = mask[:, :, None]
    image = np.concatenate([image, mask], 2)
    image = np.transpose(image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    return image, add_

def transform_valid(image, mask):
    images = []
    image = cv2.resize(image, (512, 256))
    mask = cv2.resize(mask, (512, 256))
    mask = mask[:, :, None]
    image = np.concatenate([image, mask], 2)
    raw_image = image.copy()

    image = np.transpose(raw_image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    images.append(image)

    image = np.fliplr(raw_image)
    image = np.transpose(image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    images.append(image)
    return images

def eval(model, dataLoader_valid):
    with torch.no_grad():
        model.eval()
        model.mode = 'valid'
        valid_loss, index_valid= 0, 0
        all_results = []
        all_labels = []
        for valid_data in dataLoader_valid:
            images, labels, names = valid_data
            images = images.cuda()
            labels = labels.cuda().long()
            feature, local_feat, results = data_parallel(model, images)
            model.getLoss(feature[::2], local_feat[::2], results[::2], labels)
            results = torch.sigmoid(results)
            results_zeros = (results[::2, :5004] + results[1::2, 5004:])/2
            all_results.append(results_zeros)
            all_labels.append(labels)
            b = len(labels)
            valid_loss += model.loss.data.cpu().numpy() * b
            index_valid += b
        all_results = torch.cat(all_results, 0)
        all_labels = torch.cat(all_labels, 0)
        map5s, top1s, top5s = [], [], []
        if 1:
            ts = np.linspace(0.1, 0.9, 9)
            for t in ts:
                results_t = torch.cat([all_results, torch.ones_like(all_results[:, :1]).float().cuda() * t], 1)
                all_labels[all_labels == 5004 * 2] = 5004
                top1_, top5_ = accuracy(results_t, all_labels)
                map5_ = mapk(all_labels, results_t, k=5)
                map5s.append(map5_)
                top1s.append(top1_)
                top5s.append(top5_)
            map5 = max(map5s)
            i_max = map5s.index(map5)
            top1 = top1s[i_max]
            top5 = top5s[i_max]
            best_t = ts[i_max]

        valid_loss /= index_valid
        return valid_loss, top1, top5, map5, best_t

def train(freeze=False, fold_index=1, model_name='seresnext50',min_num_class=10, checkPoint_start=0, lr=3e-4, batch_size=36):
    num_classes = 5004 * 2
    model = model_whale(num_classes=num_classes, inchannels=4, model_name=model_name).cuda()
    i = 0
    iter_smooth = 50
    iter_valid = 200
    iter_save = 200
    epoch = 0
    if freeze:
        model.freeze()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  betas=(0.9, 0.99), weight_decay=0.0002)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0002)
    resultDir = './result/{}_{}'.format(model_name, fold_index)
    ImageDir = resultDir + '/image'
    checkPoint = os.path.join(resultDir, 'checkpoint')
    os.makedirs(checkPoint, exist_ok=True)
    os.makedirs(ImageDir, exist_ok=True)
    log = Logger()
    log.open(os.path.join(resultDir, 'log_train.txt'), mode= 'a')
    log.write(' start_time :{} \n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    log.write(' batch_size :{} \n'.format(batch_size))
    # Image,Id
    data_train = pd.read_csv('./input/train_split_{}.csv'.format(fold_index))
    names_train = data_train['Image'].tolist()
    labels_train = data_train['Id'].tolist()
    data_valid = pd.read_csv('./input/valid_split_{}.csv'.format(fold_index))
    names_valid = data_valid['Image'].tolist()
    labels_valid = data_valid['Id'].tolist()
    num_data = len(names_train)
    dst_train = WhaleDataset(names_train, labels_train,mode='train',transform_train=transform_train, min_num_classes=min_num_class)
    dataloader_train = DataLoader(dst_train, shuffle=True, drop_last=True, batch_size=batch_size, num_workers=16, collate_fn=train_collate)
    print(dst_train.__len__())
    dst_valid = WhaleTestDataset(names_valid, labels_valid, mode='valid',transform=transform_valid)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=batch_size * 2, num_workers=8, collate_fn=valid_collate)
    train_loss = 0.0
    valid_loss = 0.0
    top1, top5, map5 = 0, 0, 0
    top1_train, top5_train, map5_train = 0, 0, 0
    top1_batch, top5_batch, map5_batch = 0, 0, 0

    batch_loss = 0.0
    train_loss_sum = 0
    train_top1_sum = 0
    train_map5_sum = 0
    sum = 0
    skips = []
    if not checkPoint_start == 0:
        log.write('  start from{}, l_rate ={} \n'.format(checkPoint_start, lr))
        log.write('freeze={}, batch_size={}, min_num_class={} \n'.format(freeze,batch_size, min_num_class))
        model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start)),skip=skips)
        ckp = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (checkPoint_start)))
        optimizer.load_state_dict(ckp['optimizer'])
        adjust_learning_rate(optimizer, lr)
        i = checkPoint_start
        epoch = ckp['epoch']
    log.write(
            ' rate     iter   epoch  | valid   top@1    top@5    map@5  | '
            'train    top@1    top@5    map@5 |'
            ' batch    top@1    top@5    map@5 |  time          \n')
    log.write(
            '---------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    start = timer()

    start_epoch = epoch
    best_t = 0
    cycle_epoch = 0
    while i < 10000000:
        for data in dataloader_train:
            epoch = start_epoch + (i - checkPoint_start) * 4 * batch_size/num_data
            if i % iter_valid == 0:
                valid_loss, top1, top5, map5, best_t = \
                    eval(model, dataloader_valid)
                print('\r', end='', flush=True)

                log.write(
                    '%0.5f %5.2f k %5.2f  |'
                    ' %0.3f    %0.3f    %0.3f    %0.4f    %0.4f | %0.3f    %0.3f    %0.3f | %0.3f     %0.3f    %0.3f | %s \n' % ( \
                        lr, i / 1000, epoch,
                        valid_loss, top1, top5, map5, best_t,
                        train_loss, top1_train, map5_train,
                        batch_loss, top1_batch, map5_batch,
                        time_to_str((timer() - start) / 60)))
                time.sleep(0.01)

            if i % iter_save == 0 and not i == checkPoint_start:
                torch.save(model.state_dict(), resultDir + '/checkpoint/%08d_model.pth' % (i))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter': i,
                    'epoch': epoch,
                    'best_t':best_t,
                }, resultDir + '/checkpoint/%08d_optimizer.pth' % (i))


            model.train()

            model.mode = 'train'
            images, labels = data
            images = images.cuda()
            labels = labels.cuda().long()
            global_feat, local_feat, results = data_parallel(model,images)
            model.getLoss(global_feat, local_feat, results, labels)
            batch_loss = model.loss

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
            results = torch.cat([torch.sigmoid(results), torch.ones_like(results[:, :1]).float().cuda() * 0.5], 1)
            top1_batch = accuracy(results, labels, topk=(1,))[0]
            map5_batch = mapk(labels, results, k=5)

            batch_loss = batch_loss.data.cpu().numpy()
            sum += 1
            train_loss_sum += batch_loss
            train_top1_sum += top1_batch
            train_map5_sum += map5_batch
            if (i + 1) % iter_smooth == 0:
                train_loss = train_loss_sum/sum
                top1_train = train_top1_sum/sum
                map5_train = train_map5_sum/sum
                train_loss_sum = 0
                train_top1_sum = 0
                train_map5_sum = 0
                sum = 0

            print('\r%0.5f %5.2f k %5.2f  | %0.3f    %0.3f    %0.3f    %0.4f    %0.4f | %0.3f    %0.3f    %0.3f | %0.3f     %0.3f    %0.3f | %s  %d %d' % ( \
                    lr, i / 1000, epoch,
                    valid_loss, top1, top5,map5,best_t,
                    train_loss, top1_train, map5_train,
                    batch_loss, top1_batch, map5_batch,
                    time_to_str((timer() - start) / 60), checkPoint_start, i)
                , end='', flush=True)
            i += 1
           
        pass


if __name__ == '__main__':
    if 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,5'
        freeze = False
        model_name = 'senet154'
        fold_index = 1
        min_num_class = 10
        checkPoint_start = 0
        lr = 3e-4
        batch_size = 12
        print(5005%batch_size)
        train(freeze, fold_index, model_name, min_num_class, checkPoint_start, lr, batch_size)
