### 医学影像预处理

#%% import
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import interact
import pickle
import itertools as itt
from collections import defaultdict as dfDic
from ipywidgets import interact
import copy
from matplotlib.patches import Circle

import os
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Circle
from scipy.ndimage import center_of_mass
# import nibabel as nib
# import nibabel.processing as nip
# import nibabel.orientations as nio
import json

LBLVS = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
    21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum',
    27: 'Cocc', 28: 'T13'
}

COLORS = (1/255)*np.array([
    [255,  0,  0], [  0,255,  0], [  0,  0,255], [255,255,  0], [  0,255,255],
    [255,  0,255], [255,239,213],  # Label 1-7 (C1-7)
    [  0,  0,205], [205,133, 63], [210,180,140], [102,205,170], [  0,  0,128],
    [  0,139,139], [ 46,139, 87], [255,228,225], [106, 90,205], [221,160,221],
    [233,150,122], [165, 42, 42],  # Label 8-19 (T1-12)
    [255,250,250], [147,112,219], [218,112,214], [ 75,  0,130], [255,182,193],
    [ 60,179,113], [255,235,205],  # Label 20-26 (L1-6, sacrum)
    [255,235,205], [255,228,196],  # Label 27 cocc, 28 T13,
    [218,165, 32], [  0,128,128], [188,143,143], [255,105,180],
    [255,  0,  0], [  0,255,  0], [  0,  0,255], [255,255,  0], [  0,255,255],
    [255,  0,255], [255,239,213],  # 29-39 unused
    [  0,  0,205], [205,133, 63], [210,180,140], [102,205,170], [  0,  0,128],
    [  0,139,139], [ 46,139, 87], [255,228,225], [106, 90,205], [221,160,221],
    [233,150,122],   # Label 40-50 (subregions)
    [255,250,250], [147,112,219], [218,112,214], [ 75,  0,130], [255,182,193],
    [ 60,179,113], [255,235,205], [255,105,180], [165, 42, 42], [188,143,143],
[255,235,205], [255,228,196], [218,165, 32], [  0,128,128], [30,14,255]# rest unused
    ])
CMITK = ListedColormap(COLORS)
CMITK.set_bad(color='w', alpha=0)  # set NaN to full opacity for overlay
SBWIN = Normalize(vmin=-500, vmax=1300, clip=True) # 软骨窗是-500到1300, 用于显示软骨
HBWIN = Normalize(vmin=-200, vmax=1000, clip=True) # 骨窗是-200到1000, 用于显示骨头


def fdic():
    return dfDic(fdic)


def pk2file(file, data=None):
    # if not file.endswith('.pickle' or '.pkl'):
    #     file = file+'.pickle'
    if data is not None:
        with open(file, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data

def readDcm(dcm_path):
    """
    读取DICOM图像序列,返回3D Image对象
    """
    # 读取DICOM图像序列
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)

    # 注意,此时还没有加载图像数据
    # 拼接3D图像
    image3d = reader.Execute()

    # 注意,如果图像维度<=2,则执行后image3d仍为None
    # 所以需要检验
    if image3d is None:
        print('Error: less than 3 dimensions')
        return None

    # 获得图像的元数据
    img_array = sitk.GetArrayFromImage(image3d)
    print('Image data type: ', image3d.GetPixelIDTypeAsString())
    print('Image size: ', image3d.GetSize())
    print('Image spacing: ', image3d.GetSpacing())
    print('Image origin: ', image3d.GetOrigin())
    print('Image dimension: ', image3d.GetDimension())

    return image3d
#%%
def readImgSk(filePath=None, img = None, msk = False, img3C = True, RAS=False):
    '''sitk读图
    '''
    if img is None:
        # 若为文件夹
        # if os.path.isdir(filePath):
        if filePath.endswith('/'):
            img = readDcm(filePath)
        else:
            img = sitk.ReadImage(filePath)
    if RAS:
        imgRAS = (img)
    if msk is True:
        img = sitk.GetArrayFromImage(img)
        return img.astype(np.uint8)
        # if isinstance(img,sitk.Image):
        #     img = sitk.GetArrayFromImage(img)
        #     print('img.max():',img.max())
        # if img.max()>99.:
        #     print('Warning: mask image has value > 99')
        #     img = img//10/1.0
    if isinstance(img,np.ndarray):
        img=sitk.GetImageFromArray(img)
    img_vbCT_255 = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
    # 所有像素值都被10整除

    if img3C: # 三通道无法做魔法糖
        return sitk.GetArrayFromImage(sitk.Compose([img_vbCT_255]*3)) # 通道在最后
    else:
        return img_vbCT_255
#%%
def imgRAS(image):
    """Reorients an image to standard radiology view."""

    dir = np.array(image.GetDirection()).reshape(len(image.GetSize()), -1)
    ind = np.argmax(np.abs(dir), axis=0)
    new_size = np.array(image.GetSize())[ind]
    new_spacing = np.array(image.GetSpacing())[ind]
    new_extent = new_size * new_spacing
    new_dir = dir[:, ind]

    flip = np.diag(new_dir) < 0  #
    flip_diag = flip * -1
    flip_diag[flip_diag == 0] = 1
    flip_mat = np.diag(flip_diag)

    new_origin = np.array(image.GetOrigin()) + np.matmul(new_dir, (new_extent * flip))
    new_dir = np.matmul(new_dir, flip_mat)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing.tolist())
    resample.SetSize(new_size.tolist())
    resample.SetOutputDirection(new_dir.flatten().tolist())
    resample.SetOutputOrigin(new_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    return resample.Execute(image)
def cropOtsu(image):
    ''' Otsu阈值裁图
    '''
    inside_value = 0
    outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter() # 用于计算图像的轴对齐边界框。
    label_shape_filter.Execute(sitk.OtsuThreshold(image, inside_value, outside_value))
    bounding_box = label_shape_filter.GetBoundingBox(outside_value) # 获取边界框
    print('bounding_box', bounding_box)
    return sitk.RegionOfInterest(image,
                                 bounding_box[int(len(bounding_box) / 2) :],
                                 bounding_box[0 : int(len(bounding_box) / 2)],)

def sk3C2gd(img):
    # 三通道转灰度
    if img.GetDimension() == 3:
        return (sitk.VectorIndexSelectionCast(img, 0) +
                sitk.VectorIndexSelectionCast(img, 1) +
                sitk.VectorIndexSelectionCast(img, 2)) / 3


def orientDic(img):
    return {'X': 'L-R'[img.GetDirection()[0]+1],
            'Y': 'P-A'[img.GetDirection()[4]+1],
            'Z': 'I-S'[img.GetDirection()[8]+1]}

#%%
def skShow(img,
           xyz = 'Z',
           msk = None,
           bbx = None,
           lbs = None,
           title=None,
           margin=0.05,
           dpi=96,
           fSize=None,
           cmap="gray"):
    if isinstance(img, np.ndarray):
        nda = np.copy(img)
        img = sitk.GetImageFromArray(img)
    else:
        img = copy.deepcopy(img)
        if isinstance(img[0],bool):
            nda = np.zeros(img.shape, dtype=np.uint8)
            nda[img] = 1
        nda = sitk.GetArrayFromImage(img)
    if msk is not None:
        # ndaM = np.transpose(msk, (2, 1, 0))
        # msk[msk==0]=np.nan
        ndaM = sitk.GetArrayFromImage(msk)
        ndaM[ndaM==0] = np.nan

    spacing = img.GetSpacing()
    size = img.GetSize()
    if nda.ndim == 3: # 若3维数组
        # fastest dim, either component or x
        c = nda.shape[-1] # 通道数
        if c in (3, 4): # 若通道数为3或4, 则认为是2D图像
            nda = nda[:,:,0]
    elif nda.ndim == 4: # 若4维数组
        c = nda.shape[-1]
        if not c in (3, 4): # 若通道数不为3或4, 则认为是3Dv(4D)图像, 退出
            raise RuntimeError("Unable to show 3D-vector Image")
        else:
            # 去掉最后一维
            nda = nda[:,:,:,0]
    if nda.ndim == 2: # 若2维数组
        nda = nda[np.newaxis, ...] # nda增加后的维度为3维, 且最后一维为1
        ndaM = ndaM[np.newaxis, ..., np.newaxis] if msk is not None else None
        size = size + (1,) # size增加后的维度为3维, 且最后一维为1
        spacing = 1.
    # nda.shape = shape# nda的方向为LPS
    # size = nda.shape # size为z,y,x
    print('size:',size)
    xyzSize = [int(i+1)
                for i
                in (np.array(spacing)*np.array(size))
                ]
    sInd = {'X':2, 'Y':1, 'Z':0}[xyz]
    # sDic = dfDic(fdic)
    sDic = [dict(drt=['P==>A', 'L==>R'],
                        arr = nda, # nda的方向为LP
                        arrM = ndaM if msk is not None else None,
                        x = xyzSize[0],
                        y = xyzSize[1],
                        z = size[2],
                        extent = (0, xyzSize[0], 0, xyzSize[1]) # (left, right, bottom, top)
                        ),
            dict(drt=['I==>S', 'L==>R'],
                        arr=np.transpose(nda, (1,0,2)), # nda的方向为LS
                        arrM = np.transpose(ndaM, (1,0,2)) if msk is not None else None,
                        x = xyzSize[0],
                        y = xyzSize[2],
                        z = size[1],
                        extent = (0,xyzSize[0],xyzSize[2],0) # (left, right, bottom, top)
                        ),
            dict(drt=['I==>S', 'A<==P'],
                        arr=np.transpose(nda, (2,0,1)), # nda的方向为SP
                        arrM = np.transpose(ndaM, (2,0,1)) if msk is not None else None,
                        x = xyzSize[1],
                        y = xyzSize[2],
                        z = size[0],
                        extent = (0,xyzSize[1],xyzSize[2],0) # (left, right, bottom, top)
                        )][sInd]
    def callback(axe=None):
        figsize = (1 + margin) * sDic['y'] / dpi, (1 + margin) * sDic['x'] / dpi
        fig = plt.figure(figsize=[figsize, fSize][fSize is not None], dpi=dpi)
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
        ax.imshow(sDic['arr'][axe, ...], extent=sDic['extent'], interpolation=None,  cmap=cmap) #, norm=SBWIN)
        if msk is not None:
            mArr = sDic['arrM'][axe, ...]
            ax.imshow(mArr, extent=sDic['extent'], interpolation=None, cmap=CMITK, alpha=0.3, vmin=1, vmax=64)
            if bbx is not None and xyz=='Z': # [TODO]总是定不准位置
                ls = np.unique(mArr)
                print('ls:', ls)
                for l in ls:
                    if not np.isnan(l):
                        def __bBx(mArr, label):
                            y, x = np.where(mArr == label) # x,y为msk中值为label的点的坐标
                            xy0 = np.array([np.min(x), np.min(y)])
                            xy1 = np.array([np.max(x), np.max(y)])
                            pad = np.random.randint(0, 10)
                            return xy0, xy1, pad
                        xy0, xy1, pad = __bBx(mArr, l)
                        x0, y0 = xy0
                        x1, y1 = xy1
                        lw = 2 + pad
                        z = axe
                        ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor=COLORS[int(l-1)], facecolor=(0,0,0,0), lw=lw))
        if lbs is not None:
            llbs = np.unique(lbs)
            leg = ax.legend(handles=[plt.Rectangle((0, 0), 5, 5, color=COLORS[int(lb)]) for lb in llbs], labels=[LBLVS[int(lb)] for lb in llbs], loc='center left')
            leg_height = leg.get_frame().get_height()
            fig_height = fig.get_figheight()
            if leg_height > fig_height:
                fig.set_figheight(leg_height)  # Set fig height to match legend height
    # 在图像上标注坐标轴
        ax.set_ylabel(sDic['drt'][0])
        ax.set_xlabel(sDic['drt'][1])
        # 根据颜色标lv图例

        if title:
            plt.title(title)
        return plt.show()
    interact(callback, axe=(0, sDic['z'] - 1))
#%%
def skObb(msk):
    if isinstance(msk, np.ndarray):
        msk = getMsk01(msk)
        msk = sitk.GetImageFromArray(msk)
    shpSts = sitk.LabelShapeStatisticsImageFilter()
    shpSts.ComputeOrientedBoundingBoxOn()
    shpSts.Execute(msk)
    return shpSts.GetOrientedBoundingBox(1)
# def showObbs(xyxys):
#     fig = plt.figure(figsize=(8,8))
#     ax = fig.add_subplot(111)
#     ax.imshow(labels)
#     for i in obbs.keys():
#         corners, centre = obbs[i]
#         ax.scatter(centre[1],centre[0])
#         ax.plot(corners[:,1],corners[:,0],'-')
#     plt.show()

#%%
def bBx2xy(bBx: np.ndarray) -> np.ndarray:
    xywh = np.asarray(bBx)
    dHf = len(xywh)//2
    ini, fin = xywh[:dHf],xywh[:dHf]+xywh[dHf:]
    return np.asarray(ini.tolist()+fin.tolist())

#Q: 已知xyxy1和xyxy2, iou是怎么计算出来的?
#A: 1. 计算两个框的面积, 2. 计算两个框的交集, 3. 计算iou
def getIou(xyxy1: list, xyxy2: list):
    box1_area = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])
    box2_area = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])
    interSection = max(0, min(xyxy1[2], xyxy2[2]) - max(xyxy1[0], xyxy2[0])) * max(0, min(xyxy1[3], xyxy2[3]) - max(xyxy1[1], xyxy2[1]))
    iou = interSection / (box1_area + box2_area - interSection)
    return iou

def svAllSam(sData=None, sam=True, cId=0): # 将全分的结果转为sv.Dets
    '''
    Notes:

    '''

    if isinstance(sData[0], dict):
        sortedsData = sorted(
            sData, key=lambda x: x["area"], reverse=True
        )
        xywh = np.array([mask["bbox"] for mask in sortedsData])
        masks = np.array([mask["segmentation"] for mask in sortedsData])
        # 若masks为bool型, 则转为int型
        masks = getMsk01(masks)
        confidence = np.array([mask["predicted_iou"] for mask in sortedsData])
        tracker_id = np.arange(len(sortedsData))
        class_id = np.ones(len(sortedsData))*cId
        return sv.Detections(xyxy=bBx2xy(xywh),
                            mask=masks,
                            confidence=np.array(confidence),
                            tracker_id=tracker_id,
                            class_id=class_id
                            )
    elif isinstance(sData[0], list):
        for i, data in enumerate(sData):
            if sam:
                data = svAllSam(data, cId=i)
            if i == 0:
                dets = sv.Detections(xyxy=data.xyxy,
                                    mask=data.mask,
                                    confidence=data.confidence,
                                    tracker_id=data.tracker_id,
                                    class_id=data.class_id
                                    )

            else:
                dets.xyxy=np.append(
                    dets.xyxy,data.xyxy, axis=0)
                dets.mask=np.append(
                    dets.mask,data.mask, axis=0)
                dets.confidence=np.append(
                    dets.confidence,data.confidence, axis=0)
                dets.tracker_id=np.append(
                    dets.tracker_id,data.tracker_id, axis=0)
                dets.class_id=np.append(
                    dets.class_id,data.class_id, axis=0)
        return dets

def getMsk01(mArr):
    if isinstance(mArr, sitk.Image):
        mArr = sitk.GetArrayFromImage(mArr>0)
    if isinstance(mArr[0], bool):
        mArr = np.where(mArr == True, 1, 0)
        # 若masks的不是[0,1]型, 则转为[0,1]型
    else:
        mArr = np.where(mArr>1, 1, 0)
    return mArr

def bBin(arr):
    return np.where(arr == True, 1, 0)
def bBool(arr, thr=0):
    return np.where(arr>thr, True, False)
#%%
class GetMskBbx:
    ''' 罩机
        para:
            msk: 罩
            pad: 罩最厚
        retun:
            Dic: 罩字典
                'bBxes': Gt_xyxy, pad, z,
                'pBbx': pad_xyxy,
                'lbLy': label, layer,
                'lvBbx': lv_xyxy,
                'bBx': bBx_xyxy(optional)
    '''
    def __init__(self, msk, pad=10):
        """Initializes the GetBbx class."""
        if isinstance(msk, sitk.Image):
            self.mArr = sitk.GetArrayFromImage(msk)
        else:
            self.mArr = msk
        self.pad = pad
        self.Dic = dfDic(fdic)
        labels = np.unique(self.mArr)
        if len(labels) > 2:
            self.labels = labels[1:]
            self.get_3d_bbox()
        elif len(labels) == 2:
            label = label[1]
            level = LBLVS[label]
            if len(self.mArr)==2:
                bBx, pBbx = self.get_2d_bbox(self.mArr[z],label)
                self.Dic['bBx'][level] = bBx
                self.Dic['pBbx'][level] = pBbx
            else:
                self.get_3d_bbox()
        else:
            raise ValueError('No labels found in mask.')
    def get_2d_bbox(self, arr2D, label, one=False):
        """Gets the 2D bounding box for a label."""
        y, x = np.where(arr2D == label)
        bBx = np.array([np.min(x), np.min(y), np.max(x), np.max(y)])
        # if self.pad is not None:
        h, w = arr2D.shape
        pBbx = np.asarray([
            max(0, self.Dic['bBx'][0] - np.random.randint(0, self.pad)),
            max(0, self.Dic['bBx'][1] - np.random.randint(0, self.pad)),
            min(w, self.Dic['bBx'][2] + np.random.randint(0, self.pad)),
            min(h, self.Dic['bBx'][3] + np.random.randint(0, self.pad))
        ])
        return bBx, pBbx
    def get_3d_bbox(self):
        """Gets the 3D bounding box for a label."""
        for label in self.labels:
            z, y, x = np.where(self.mArr == label)
            z0, z1 = np.min(z), np.max(z)
            xyz0 = np.array([np.min(x), np.min(y), z0])
            xyz1 = np.array([np.max(x), np.max(y), z1])
            level = LBLVS[label]
            self.Dic['lvBbx'][level]=np.array([xyz0, xyz1])
            print(level)
            for z in range(xyz0[2], xyz1[2] + 1):
                try:
                    bBx, pBbx = self.get_2d_bbox(self.mArr[z],label)
                    self.Dic['bBx'][level][z] = bBx
                    self.Dic['pBbx'][level][z] = pBbx
                except:
                    print(f'No 2D bbox found for slice {z-xyz1[2]}')
                    self.Dic['bBx'][level][z] = None
                    self.Dic['pBbx'][level][z] = None
#%%
def dataIso(img_nib, msk_nib, ctdList, spc=(1,1,1), axes='IPL'):
    # Resample and Reorient data
    if spc is None:
        spc = img_nib.header.get_zooms()
        mSpc = msk_nib.header.get_zooms()
        img_iso = img_nib
        msk_iso = [msk_nib, resample_mask_to(msk_nib, img_iso)][mSpc==spc]
        ctds = [ctdList, rescale_centroids(ctdList, img_nib, spc)][mSpc==spc]
    else:
        img_iso = resample_nib(img_nib, voxel_spacing=spc, order=3)
        msk_iso = resample_nib(msk_nib, voxel_spacing=spc, order=0)
        ctds = rescale_centroids(ctdList, img_nib, spc) # 质心缩放
    img_iso = reorient_to(img_iso, axcodes_to=axes)
    msk_iso = reorient_to(msk_iso, axcodes_to=axes)
    ctds = reorient_centroids_to(ctds, img_iso)
    return img_iso, msk_iso, ctds
#%%
def samShow(img=None, masks=None, bxCapt=['',''], points=None,
            boxes=None, fSize=(10,10), dpi = 66, pad=20, opacity=0.4):
    def __showMsk(masks=masks, ax=None, opacity=opacity):
        labels = np.unique(masks)[1:]
        print(labels)
        # if labels > 0: # or not np.isnan(labels):
        for lb in labels:
            level = LBLVS[lb]
            vClr = COLORS[int(lb)]
            mask = np.array(masks==lb)
            h, w = mask.shape[-2:]
            vClr = np.array(vClr.tolist() + [opacity])
            mask_image = mask.reshape(h, w, 1) * vClr.reshape(1, 1, -1)
            ax.imshow(mask_image)

            # ax.patches.clear()
            yInd, xInd = np.where(mask==1)
            # print(len(yInd),len(xInd))
            x0, x1 = np.min(xInd), np.max(xInd)
            y0, y1 = np.min(yInd), np.max(yInd)
            ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor=vClr, facecolor=(0,0,0,0), lw=3))
            if pad is not None:
                h, w = mask.shape
                xP0 = max(0, x0 - np.random.randint(0, pad))
                xP1 = min(w, x1 + np.random.randint(0, pad))
                yP0 = max(0, y0 - np.random.randint(0, pad))
                yP1 = min(h, y1 + np.random.randint(0, pad))
                ax.add_patch(plt.Rectangle((xP0,yP0),xP1-xP0,yP1-yP0,edgecolor=vClr,facecolor=(0,0,0,0), lw=2))
            locLb = level + ' ' + bxCapt[0]
            iouLb = f' {bxCapt[1]}'
            plt.text(x0, y0, locLb+iouLb, fontsize=12, color='white',backgroundcolor=vClr)
            plt.text(x0, y0, locLb+iouLb, fontsize=12, color='white',backgroundcolor=vClr)
            # plt.text(x1, y1, 'A\n R', fontsize=12, color='white',backgroundcolor=vClr)

    # def __showPs(coords, labels, ax, marker_size=375):
    #     pos_points = coords[labels==1]
    #     neg_points = coords[labels==0]
    #     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    #     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    # def __showBox(box, ax):
    #     x0, y0 = box[0], box[1]
    #     w, h = box[2] - box[0], box[3] - box[1]
    #     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    if fSize is None:
        if img is not None:
            h, w = img.shape[1:3]
        elif masks is not None:
            h, w = masks.shape[1:3]
        fSize = (w/dpi, h/dpi)
    else:
        h,w = np.asarray(fSize)*dpi

    def callback(zI_to_S=None):
        plt.figure(figsize=fSize, dpi=dpi)
        # plt.gca().clear()
        if zI_to_S is None:
            if img is not None:
                plt.imshow(img, cmap='gray')
            if masks is not None:
                # plt.gca().patches.clear()
                __showMsk(masks, plt.gca())
            plt.imshow(img, cmap='gray')
        else:
            if img is not None:
                plt.imshow(img[zI_to_S,...], cmap='gray')
            if masks is not None:
                __showMsk(masks[zI_to_S,...], plt.gca()) # gca: get current axis
        # if points is not None:
        #     __showPs(points, labels, plt.gca())
        # if boxes is not None:
        #     for box in [boxes,[boxes]][int(len(boxes)==1)]:
        #         __showBox(box, plt.gca())
        plt.text(0,0,'O L>>>R\nP\nV\nV\nA', ha='left',va='top',fontsize=16, color='white')
        plt.axis('off')
        plt.show()

        return plt

    imgs = [masks,img][img is not None]
    print(imgs.shape)
    if imgs.ndim == 2:
        callback()
    elif imgs.ndim in[3,4]:
        if imgs.shape[2] in [1,3]:
            # imgs添加一个维度到第一个位置且值为2
            callback()
        else:
            zAxis = imgs.shape[0]-1
            # print(imgs.shape)
            interact(callback, zI_to_S=(0, zAxis))


#%%
#%%
def obb2d(mask):
    x, y = np.where(mask > 0)
    mkArr = np.stack((x, y), axis=1)
    cov = np.cov(mkArr, y = None,rowvar = 0,bias = 1)
    _, vect = np.linalg.eig(cov)
    tvect = np.transpose(vect)
    points_r = np.dot(mkArr, np.linalg.inv(tvect))

    co_min = np.min(points_r, axis=0)
    co_max = np.max(points_r, axis=0)

    xmin, xmax = co_min[0], co_max[0]
    ymin, ymax = co_min[1], co_max[1]

    x_x = xmax - xmin
    y_y = ymax - ymin

    xdif = (x_x) * 0.5
    ydif = (y_y) * 0.5

    xDim = np.linalg.norm(x_x)
    yDim = np.linalg.norm(y_y)

    cx = xmin + xdif
    cy = ymin + ydif

    corners = np.array([
                        [cx - xdif, cy - ydif],
                        [cx - xdif, cy + ydif],
                        [cx + xdif, cy - ydif],
                        [cx + xdif, cy + ydif],
                        ])

    return corners
#%%
#%%
def end():

    pass