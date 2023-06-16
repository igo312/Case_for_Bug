class ATSSHead_spbn_s_yolo(nn.Module):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 pred_kernel_size: int = 3,
                 stacked_convs: int = 1,
                 tiny=False,
                 feat_channels: int = 128):

        self.decode_mean = [0., 0., 0., 0.]
        self.decode_std = [0.1, 0.1, 0.2, 0.2]
        
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.in_channels = in_channels
        self.tiny = tiny
        super().__init__()
        self.sampling = False
        self.relu = nn.ReLU(inplace=True)
        self.cls_reg_convs = nn.ModuleList()
        self.prior_generator_strides = [(8, 8), (16, 16), (32, 32)]

        self.num_anchors = 1
        self.cls_out_channels = num_classes
        self.num_base_priors = 1
        self.strides = [8, 16, 32]
        # featmap_sizes = [[36, 60], [18, 30], [9, 15]]
        featmap_sizes = [[48, 80], [24, 40], [12, 20]]
        self.anchors = self.grid_prios(featmap_sizes, self.strides)
        self.feat_num = len(featmap_sizes)
        
        for n in range(len(self.prior_generator_strides)):
            cls_reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                if self.tiny:
                    cls_reg_convs.append(
                        nn.Conv2d(
                            in_channels=chn,
                            out_channels=self.num_anchors * self.cls_out_channels + self.num_base_priors * 5,
                            kernel_size=3,
                            stride=1,
                            padding=1))
                else:
                    cls_reg_convs.append(
                        head_ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1))
            self.cls_reg_convs.append(cls_reg_convs)

        pred_pad_size = self.pred_kernel_size // 2
        
        if self.tiny:
            self.atss_cls = (nn.Identity())
        else:
            self.atss_cls = (nn.Conv2d(
                self.feat_channels,
                self.num_anchors * self.cls_out_channels,
                self.pred_kernel_size,
                padding=pred_pad_size))
        
        if self.tiny:
            self.atss_cls_reg_centerness = (nn.Identity())
        else:
            self.atss_cls_reg_centerness = (nn.Conv2d(
                self.feat_channels,
                self.num_anchors * self.cls_out_channels + self.num_base_priors * 5,
                self.pred_kernel_size,
                padding=pred_pad_size))
        
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator_strides])
        share_conv = True
        if share_conv:
            for n in range(len(self.prior_generator_strides)):
                for i in range(self.stacked_convs):
                    if self.tiny:
                        self.cls_reg_convs[n][i] = self.cls_reg_convs[0][i]
                    else:
                        self.cls_reg_convs[n][i].conv = self.cls_reg_convs[0][i].conv

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        
        cls_score_all, bbox_pred_all, centerness_all = [], [], []
        for idx, (feat, stride) in enumerate(
                zip(x, self.prior_generator_strides)):

            for cls_layer in self.cls_reg_convs[idx]:
                feat = cls_layer(feat)
            
            feat = self.atss_cls_reg_centerness(feat)
            feat0, feat1, feat2 = torch.split(feat, [self.num_anchors * self.cls_out_channels, self.num_base_priors * 4, self.num_base_priors * 1], 1)
            cls_score = feat0
            bbox_pred = self.scales[idx](feat1).float()
            centerness = feat2
            if idx == 0:
                self.inp_height, self.inp_width = bbox_pred.shape[2:]
                self.inp_height, self.inp_width = self.inp_height*self.strides[0], self.inp_width*self.strides[0]
            cls_score_all.append(flatten_pred(cls_score))
            bbox_pred_all.append(flatten_pred(bbox_pred))
            centerness_all.append(flatten_pred(centerness))
            
        flatten_anchors = self.anchors.to(cls_score_all[0].device) # (N,4)
        cls_score_all = torch.concat(cls_score_all, dim=1)
        bbox_pred_all = torch.concat(bbox_pred_all, dim=1) # (N,4)
        centerness_all = torch.concat(centerness_all, dim=1)
        bbox_decoded_all = self.decode_bbox(bbox_pred_all, flatten_anchors)
        bbox_decoded_all.unsqueeze_(dim=2)
        
        return cls_score_all.sigmoid()*centerness_all.sigmoid(), bbox_decoded_all
    
    def grid_prios(self, featmap_sizes, strides=[8, 16, 32], base_anchor=[[-1.5, -1.5, 1.5, 1.5]]):
        anchors = []
        base_anchor = torch.tensor(np.asarray(base_anchor))
        for i in range(len(featmap_sizes)):
            feat_h, feat_w = featmap_sizes[i]
            stride = strides[i]
            shift_x, shift_y = torch.arange(0, feat_w) * stride, torch.arange(0, feat_h) * stride
            # create mesh grid 
            shift_xx, shift_yy = shift_x.repeat(shift_y.shape[0]), shift_y.view(-1, 1).repeat(1, shift_x.shape[0]).view(-1)
            shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
            # create anchor 
            anchor = base_anchor[None, :, :]*stride + shifts[:, None, :]
            anchor = anchor.view(-1, 4)
            anchors.append(anchor)
        return torch.concat(anchors, dim=0)
    
    def decode_bbox(self, deltas, anchors):
        mean = self.decode_mean 
        std = self.decode_std
        mean = deltas.new_tensor(mean).view(1, -1)
        std = deltas.new_tensor(std).view(1, -1)
        denorm_deltas = deltas * std + mean 

        dxy = denorm_deltas[..., :2]
        dwh = denorm_deltas[..., 2:]

        axy = ((anchors[:, :2] + anchors[:, 2:]) * 0.5)[None, ...]
        awh = (anchors[:, 2:] - anchors[:, :2])[None, ...]

        dxy_wh = awh * dxy 
        dwh = dwh.clamp(min=-2048, max=2048)

        gxy = axy + dxy_wh
        gwh = awh * dwh.exp()
        x1y1 = gxy - (gwh * 0.5)
        x2y2 = gxy + (gwh * 0.5)
        bboxes = torch.cat([x1y1, x2y2], dim=-1)
        bboxes[..., 0::2].clamp_(min=0, max=self.inp_width)
        bboxes[..., 1::2].clamp_(min=0, max=self.inp_height)
        # TODO debug here 
        # bboxes = bboxes.reshape(num_bboxes, -1)
        return bboxes 


def flatten_pred(pred):
    batch_size, pred_elements, w, h = pred.shape
    pred = pred.reshape(batch_size, pred_elements, w*h)
    pred = torch.permute(pred, (0,2,1)).contiguous()
    return pred
