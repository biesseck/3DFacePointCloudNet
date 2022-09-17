import sys

from pointnet2.train.train_triplet import *


def main():
    model = Pointnet(input_channels=3, use_xyz=True)
    model.cuda()
    # 512 is dimension of feature
    classifier = {
        'MCP': layer.MarginCosineProduct(512, args.num_class).cuda(),
        'AL': layer.AngleLinear(512, args.num_class).cuda(),
        'L': torch.nn.Linear(512, args.num_class, bias=False).cuda()
    }[args.classifier_type]

    criterion = nn.TripletMarginLoss(margin=0.5, p=2)
    optimizer = optim.Adam(
        [{'params': model.parameters()}, {'params': classifier.parameters()}],
        lr=lr, weight_decay=args.weight_decay
    )

    model.eval()
    optimizer.zero_grad()

    # g_feature = torch.zeros((len(gfile_list), 1, 512))  # original
    # p_feature = torch.zeros((len(pfile_list), 1, 512))  # original
    g_feature = torch.zeros((1, 1, 512))                  # BERNARDO
    p_feature = torch.zeros((1, 1, 512))                  # BERNARDO

    # fname = '/home/bjgbiesseck/GitHub/MICA/demo/output_TESTE/carell/mesh.obj'
    fname = '/home/bjgbiesseck/GitHub/MICA/demo/output_TESTE/carell/mesh.ply'
    # fname = '/home/bjgbiesseck/GitHub/MICA/demo/output_TESTE/carell/kpt7.npy'

    print('Loading file:', fname)
    input = loadBCNFile(fname)
    input = input.unsqueeze(0).contiguous()
    input = input.to("cuda", non_blocking=True)
    feat = model(input)  # 1x512
    g_feature[i, :, :] = feat.cpu()  # 105x1x512


if __name__ == '__main__':
    # sys.argv += ['-epochs', '100']
    # print('__main__(): sys.argv=', sys.argv)

    args = parse_args()
    # print('__main__(): args=', args)

    main()
