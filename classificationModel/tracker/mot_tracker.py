from turtle import update
from numba.cuda.simulator.api import detect
import numpy as np
from numba import jit
from collections import OrderedDict, deque
import itertools
import cv2
from numpy.lib.type_check import imag

from utils.nms_wrapper import nms_detections
from utils.log import logger
from utils.overlap import cutOverLap

from tracker import matching
#from utils.kalman_filter import KalmanFilter
from utils.particle_filter import ParticleFilter
from models.classification.classifier import PatchClassifier
from models.reid import load_reid_model, extract_reid_features


from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):

    def __init__(self, tlwh, max_n_features=100):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)

        self.min_score = 0.3
        self.particle_filter = None
        self.mean = None
        self.particles = None
        self.is_activated = False

        self.tlwhs = None
        
        self.max_n_features = max_n_features
        self.curr_feature = None
        self.last_feature = None
        self.features = deque([], maxlen=self.max_n_features)
        self.tracklet_len = 0
        self.time_by_tracking = 0
        # classification
        self.overlay = []
        self.belong = []

    def set_feature(self, feature):
        if feature is None:
            return False
        self.features.append(feature)
        self.curr_feature = feature
        self.last_feature = feature
        # self._p_feature = 0
        return True

    def predict(self, w, h):
        self.particles = self.particle_filter.transition(self.particles, w, h)

    # def activate(self, kalman_filter, frame_id, image):
    def activate(self, particle_filter, frame_id):
        """Start a new tracklet"""
        self.particle_filter = particle_filter
        self.track_id = self.next_id()
        self.particles = self.particle_filter.initiate(self._tlwh)
        self.mean = self.tlwh_to_xyah(self._tlwh)
        del self._tlwh
        self.state = TrackState.Tracked
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(new_track.tlwh))
        # self.mean, self.covariance = self.kalman_filter.update(
        #   self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        # )

        # 重新初始化粒子
        self.particles = self.particle_filter.initiate(new_track._tlwh)
        self.mean = self.tlwh_to_xyah(new_track._tlwh)
        self.time_since_update = 0
        self.time_by_tracking = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        # self.set_feature(new_track.curr_feature)

    def update(self, classfier, frame_id, image, reid_model):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.time_since_update = 0
        self.tracklet_len += 1

        #更新粒子权重
        flag,self.particles = self.particle_filter.updateweight(
            self.particles, classfier, reid_model, self.curr_feature, image, self.min_score)

        if flag:
            self.state = TrackState.Tracked
            self.particles = self.particle_filter.resample(self.particles)
            self.is_activated = True
        else:
            self.state = TrackState.Lost
            self.is_activated = False

        x = self.particles[0].x
        y = self.particles[0].y
        width = self.particles[0].width
        height = self.particles[0].height
        a = width/height
        self.mean = np.asarray([x,y,a,height] ,dtype = float)

    @property
    @jit
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean.copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    @jit
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        # print('ret=',)
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    @jit
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def tracklet_score(self):
        # score = (1 - np.exp(-0.6 * self.hit_streak)) * np.exp(-0.03 * self.time_by_tracking)
        score = self.particles[0].w+1
        return score
    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class OnlineTracker(object):

    def __init__(self, min_cls_score=0.4, min_ap_dist=0.64, max_time_lost=30, use_tracking=True, use_refind=True):

        self.min_cls_score = min_cls_score
        self.min_ap_dist = min_ap_dist
        self.max_time_lost = max_time_lost

        #self.kalman_filter = KalmanFilter()
        self.particle_filter = ParticleFilter()

        self.tracked_stracks = []   # type: list[STrack]
        self.lost_stracks = []      # type: list[STrack]
        self.removed_stracks = []   # type: list[STrack]

        self.use_refind = use_refind
        self.use_tracking = use_tracking
        self.classifier = PatchClassifier()
        self.reid_model = load_reid_model()

        self.frame_id = 0
    def update(self, image, tlwhs, det_scores):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        h, w, _ = image.shape
        self.classifier.update(image)

        nms_tlbrs = []
        scores = []
        index = []
        """step 1: prediction"""
        for i,strack in enumerate(self.tracked_stracks):
            strack.predict(w, h)
            strack.update(self.classifier,self.frame_id,image,self.reid_model)
            if strack.state == TrackState.Tracked:
                scores.append(strack.tracklet_score())
                nms_tlbrs.append(strack.tlbr)
                index.append(i)
            else:
                lost_stracks.append(strack)
        '''step2:un match'''
        n_tracked = len(scores)
        det_tlbrs = tlwhs.copy()
        det_tlbrs[:,2:] += det_tlbrs[:,:2]
        nms_tlbrs.extend(det_tlbrs)
        scores.extend(det_scores)
        scores = np.asarray(scores,dtype=np.float)
        print('len(det_tlbrs)=',len(det_tlbrs))
        print('len(scores )=',len(scores))
        if len(nms_tlbrs)>0:
            keep = nms_detections(nms_tlbrs,scores,nms_thresh=0.3)
            tracked_keep = [i for i in keep if (i>=0 and i<n_tracked)]
            #detect_keep = [i for i in keep if i>=n_tracked]
            print('index=',index)
            lost_indexs = []
            for i in range(n_tracked):
                if i not in tracked_keep:
                    lost_indexs.append(index[i])
            #index = [inde for i in range(n_tracked) if i not in tracked_keep]             #非极大值抑制中淘汰的被跟踪的轨迹索引
            #print('after_index=',index)
            for ind in lost_indexs:
                self.tracked_stracks[ind].mark_removed()
                removed_stracks.append(self.tracked_stracks[ind])
            nms_tlbrs_save = [nms_tlbrs[i] for i in keep if i>= n_tracked]
            nms_tlwhs_save = []
            for itlbr in nms_tlbrs_save:
                itlbr[2:] -= itlbr[:2]
                nms_tlwhs_save.append(itlbr)
            detect_tracks = [STrack(tlwh) for tlwh in nms_tlwhs_save]

            '''step3:update feature'''
            tracked_tracks = [track for track in self.tracked_stracks if track.state==TrackState.Tracked]
            over_tlwhs = [track.tlwh for track in itertools.chain(tracked_tracks,detect_tracks)]
            cut_tlbrs = cutOverLap(over_tlwhs,image)
            #删除较小的裁剪框
            cond1 = cut_tlbrs[:,0]+5<cut_tlbrs[:,2]
            cond2 = cut_tlbrs[:,1]+10<cut_tlbrs[:,3]
            cut_indexs = np.where([cond1[i] and cond2[i] for i in range(len(cond1))])[0]
            cut_tlbrs_true = cut_tlbrs[cut_indexs]
            update_tracks = []
            for i,update_track in enumerate(itertools.chain(tracked_tracks,detect_tracks)):
                if i in cut_indexs:
                    update_tracks.append(update_track)
                else:
                    if update_track.curr_feature is None:
                        update_track.mark_removed()
                        removed_stracks.append(update_track)
                    else:
                        update_track.mark_lost()
                        lost_stracks.append(update_track)
            cut_features = extract_reid_features(self.reid_model,image,cut_tlbrs_true)
            cut_features = cut_features.cpu().numpy()
            for track,feature in zip(update_tracks,cut_features):
                track.set_feature(feature)
        else:
                detect_tracks = []
                
                '''
                over_image = image.copy()
                for itlwh in over_tlwhs:
                    tlwh = list(map(int,itlwh))
                    x1,y1,w,h = tlwh
                    x2 = x1+w
                    y2 = y1+h
                    cv2.rectangle(over_image,(x1,y1),(x2,y2),(0,255,0),2,8,0)
                cv2.imshow('over_image ',over_image )
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                #显示剪切后的图像
                cut_img = image.copy()
                for itlbr in cut_tlbrs:
                    tlbr_ = list(map(int,itlbr))
                    x1,y1,x2,y2 = tlbr_
                    if(x1==x2) or (y1==y2):
                        print('tlbr=',itlbr)
                    cv2.rectangle(cut_img,tlbr_[:2],tlbr_[2:],(0,255,0),2,8,0)
                cv2.imshow('cut_img ',cut_img )
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''
            
        '''step4:match lost'''
        #print(len(self.lost_stracks),len(detect_tracks))
        detect_tracks = [track for track in detect_tracks if track.state==TrackState.New]
        dists = matching.nearest_reid_distance(self.lost_stracks,detect_tracks,metric='euclidean')
        matches,u_lost,u_detection = matching.linear_assignment(dists,thresh=self.min_ap_dist)
        for itracked,idet in matches:
            self.lost_stracks[itracked].re_activate(detect_tracks[idet],self.frame_id,new_id= not self.use_refind)
            refind_stracks.append(self.lost_stracks[itracked])

        '''step5:init new stracks'''
        for inew in u_detection:
            track = detect_tracks[inew]
            track.activate(self.particle_filter,self.frame_id)
            activated_starcks.append(track)
        '''step6: update state'''
        for track in self.lost_stracks:
            if self.frame_id-track.end_frame>self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.lost_stracks = [t for t in self.lost_stracks if t.state==TrackState.Lost]
        self.tracked_stracks.extend(activated_starcks)
        self.tracked_stracks.extend(refind_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.removed_stracks.extend(removed_stracks)

        # get scores of lost tracks
        rois = np.asarray(
            [t.tlbr for t in self.lost_stracks], dtype=np.float32)
        lost_cls_scores = self.classifier.predict(rois)
        out_lost_stracks = [t for i, t in enumerate(self.lost_stracks)
                            if lost_cls_scores[i] > 0.3 and self.frame_id - t.end_frame <= 4]
        output_tracked_stracks = [
            track for track in self.tracked_stracks if track.is_activated]

        output_stracks = output_tracked_stracks + out_lost_stracks

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format(
            [track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format(
            [track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format(
            [track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format(
            [track.track_id for track in removed_stracks]))

        return output_stracks
