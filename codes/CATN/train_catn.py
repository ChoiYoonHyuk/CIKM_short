import re
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import CATN


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)

    return string.strip().lower()


def pre_processing(s_data, s_dict, t_data, t_dict, w_embed, batch_size, device):
    u_embed, i_embed, aux_embed, label = [], [], [], []
    limit = 500

    for idx in range(batch_size):
        u, i, rat = s_data[0][idx], s_data[1][idx], s_data[2][idx]

        u_rev, i_rev, aux_rev = [], [], []

        reviews = s_dict[u]
        for review in reviews:
            if review[0] != i:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        u_rev.append(rev)
                    except KeyError:
                        continue

        reviews = s_dict[i]
        for review in reviews:
            if review[0] != u:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        i_rev.append(rev)
                    except KeyError:
                        continue

        if u in t_dict:
            reviews = t_dict[u]
            for review in reviews:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        aux_rev.append(rev)
                    except KeyError:
                        continue
        else:
            reviews = s_dict[u]
            for review in reviews:
                if review[0] != i:
                    review = review[1].split(' ')
                    for rev in review:
                        try:
                            rev = clean_str(rev)
                            rev = w_embed[rev]
                            u_rev.append(rev)
                        except KeyError:
                            continue

        if len(u_rev) > limit:
            u_rev = u_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(u_rev)
            for p in range(pend):
                u_rev.append(lis)

        if len(i_rev) > limit:
            i_rev = i_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(i_rev)
            for p in range(pend):
                i_rev.append(lis)

        if len(aux_rev) > limit:
            aux_rev = aux_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(aux_rev)
            for p in range(pend):
                aux_rev.append(lis)

        u_embed.append(u_rev)
        i_embed.append(i_rev)
        aux_embed.append(aux_rev)
        label.append([rat])

    for idx in range(batch_size):
        u, i, rat = t_data[0][idx], t_data[1][idx], t_data[2][idx]

        u_rev, i_rev, aux_rev = [], [], []

        reviews = t_dict[u]
        for review in reviews:
            if review[0] != i:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        u_rev.append(rev)
                    except KeyError:
                        continue

        reviews = t_dict[i]
        for review in reviews:
            if review[0] != u:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        i_rev.append(rev)
                    except KeyError:
                        continue

        if u in s_dict:
            reviews = t_dict[u]
            for review in reviews:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        aux_rev.append(rev)
                    except KeyError:
                        continue
        else:
            reviews = t_dict[u]
            for review in reviews:
                if review[0] != i:
                    review = review[1].split(' ')
                    for rev in review:
                        try:
                            rev = clean_str(rev)
                            rev = w_embed[rev]
                            u_rev.append(rev)
                        except KeyError:
                            continue

        if len(u_rev) > limit:
            u_rev = u_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(u_rev)
            for p in range(pend):
                u_rev.append(lis)

        if len(i_rev) > limit:
            i_rev = i_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(i_rev)
            for p in range(pend):
                i_rev.append(lis)

        if len(aux_rev) > limit:
            aux_rev = aux_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(aux_rev)
            for p in range(pend):
                aux_rev.append(lis)

        u_embed.append(u_rev)
        i_embed.append(i_rev)
        aux_embed.append(aux_rev)
        label.append([rat])

    u_embed = torch.tensor(u_embed, requires_grad=True).view(batch_size * 2, 1, 500, 100).to(device)
    i_embed = torch.tensor(i_embed, requires_grad=True).view(batch_size * 2, 1, 500, 100).to(device)
    aux_embed = torch.tensor(aux_embed, requires_grad=True).view(batch_size * 2, 1, 500, 100).to(device)
    label = torch.FloatTensor(label).to(device)

    return u_embed, i_embed, aux_embed, label


def ans_processing(s_dict, t_data, t_dict, w_embed, device):
    batch_size = 32
    # Return embedded vector [user, item, rev_ans, rat]
    u_embed, i_embed, aux_embed, label = [], [], [], []
    limit = 500

    for idx in range(batch_size):
        u, i, rat = t_data[0][idx], t_data[1][idx], t_data[2][idx]

        u_rev, i_rev, aux_rev = [], [], []

        reviews = t_dict[u]
        for review in reviews:
            if review[0] != i:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        u_rev.append(rev)
                    except KeyError:
                        continue

        reviews = t_dict[i]
        for review in reviews:
            if review[0] != u:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        i_rev.append(rev)
                    except KeyError:
                        continue

        if u in s_dict:
            reviews = t_dict[u]
            for review in reviews:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        aux_rev.append(rev)
                    except KeyError:
                        continue
        else:
            reviews = t_dict[u]
            for review in reviews:
                if review[0] != i:
                    review = review[1].split(' ')
                    for rev in review:
                        try:
                            rev = clean_str(rev)
                            rev = w_embed[rev]
                            u_rev.append(rev)
                        except KeyError:
                            continue

        if len(u_rev) > limit:
            u_rev = u_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(u_rev)
            for p in range(pend):
                u_rev.append(lis)

        if len(i_rev) > limit:
            i_rev = i_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(i_rev)
            for p in range(pend):
                i_rev.append(lis)

        if len(aux_rev) > limit:
            aux_rev = aux_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(aux_rev)
            for p in range(pend):
                aux_rev.append(lis)

        u_embed.append(u_rev)
        i_embed.append(i_rev)
        aux_embed.append(aux_rev)
        label.append([rat])

    u_embed = torch.tensor(u_embed, requires_grad=True).view(batch_size, 1, 500, 100).to(device)
    i_embed = torch.tensor(i_embed, requires_grad=True).view(batch_size, 1, 500, 100).to(device)
    aux_embed = torch.tensor(aux_embed, requires_grad=True).view(batch_size, 1, 500, 100).to(device)
    label = torch.FloatTensor(label).to(device)

    return u_embed, i_embed, aux_embed, label


def valid(s_dict, v_data, t_dict, w_embed, save, t_data, write_file, device):
    batch_size = 32
    model = CATN()
    model.load_state_dict(torch.load(save, map_location=device))
    model.to(device)
    model.eval()

    text_conv = model.text_conv
    agc = model.aspect_gate_control
    aat = model.t_aspect_attention
    cls = model.t_classifier

    criterion = nn.MSELoss()

    v_batch = DataLoader(v_data, batch_size=batch_size, shuffle=True, num_workers=2)
    v_loss, idx = 0, 0

    for v_data in tqdm(v_batch, leave=False):
        if len(v_data[0]) != batch_size:
            continue
        u_rev, i_rev, aux_rev, label = ans_processing(s_dict, v_data, t_dict, w_embed, device)

        with torch.no_grad():
            u_rev = text_conv(u_rev).transpose(1, 3)
            i_rev = text_conv(i_rev).transpose(1, 3)
            aux_rev = text_conv(aux_rev).transpose(1, 3)
            # 64, 1, 496, 32

            u_rev = agc(u_rev).transpose(2, 3)
            i_rev = agc(i_rev).transpose(2, 3)
            aux_rev = agc(aux_rev).transpose(2, 3)
            # 64, 1, 16, 496

            u_rev = aat(u_rev).squeeze()
            aux_rev = aat(aux_rev).squeeze()
            # 64, 16, 16

            i_rev = aat(i_rev).squeeze()
            u_rev = (u_rev + aux_rev) / 2

            u_rev = u_rev.reshape(32, 256, 1).squeeze()
            i_rev = i_rev.reshape(32, 256, 1).squeeze()

            out = torch.cat((u_rev, i_rev), 1)

            out = cls(out)
            print(out)

            v_loss += criterion(out, label)

        idx += 1
    v_loss = v_loss / idx

    t_batch = DataLoader(t_data, batch_size=batch_size, shuffle=True, num_workers=2)
    t_loss, idx = 0, 0

    for t_data in tqdm(t_batch, leave=False):
        if len(t_data[0]) != batch_size:
            continue
        u_rev, i_rev, aux_rev, label = ans_processing(s_dict, t_data, t_dict, w_embed, device)

        with torch.no_grad():
            u_rev = text_conv(u_rev).transpose(1, 3)
            i_rev = text_conv(i_rev).transpose(1, 3)
            aux_rev = text_conv(aux_rev).transpose(1, 3)
            # 64, 1, 496, 32

            u_rev = agc(u_rev).transpose(2, 3)
            i_rev = agc(i_rev).transpose(2, 3)
            aux_rev = agc(aux_rev).transpose(2, 3)
            # 64, 1, 16, 496

            u_rev = aat(u_rev).squeeze()
            aux_rev = aat(aux_rev).squeeze()
            # 64, 16, 16

            i_rev = aat(i_rev).squeeze()
            u_rev = (u_rev + aux_rev) / 2

            u_rev = u_rev.reshape(32, 256, 1).squeeze()
            i_rev = i_rev.reshape(32, 256, 1).squeeze()

            out = torch.cat((u_rev, i_rev), 1)

            out = cls(out)

            t_loss += criterion(out, label)
        idx += 1

    t_loss = t_loss / idx

    print('Validation MSE Loss: %.4f, Target MSE Loss: %.4f' % (v_loss, t_loss))

    #w = open(write_file, 'a')
    #w.write('%.6f %.6f\n' % (v_loss, t_loss))


def learning(s_data, s_dict, t_data, t_dict, w_embed, save, idx, device):
    # Model
    print('Learning ... \n')
    model = CATN()
    if idx == 1:
        model.load_state_dict(torch.load(save, map_location=device))
    model.to(device)
    model.train()

    criterion = nn.MSELoss()

    text_conv = model.text_conv
    agc = model.aspect_gate_control
    s_aat = model.s_aspect_attention
    t_aat = model.t_aspect_attention
    s_cls = model.s_classifier
    t_cls = model.t_classifier

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Make batch
    batch_size = 32
    s_batch = DataLoader(s_data, batch_size=batch_size, shuffle=True, num_workers=2)
    t_batch = DataLoader(t_data, batch_size=batch_size, shuffle=True, num_workers=2)

    batch_data, zip_size = zip(s_batch, t_batch), min(len(s_batch), len(t_batch))

    for source_x, target_x in tqdm(batch_data, leave=False, total=zip_size):
        # Pre processing
        if len(source_x[0]) != batch_size or len(target_x[0]) != batch_size:
            continue
        u_rev, i_rev, aux_rev, label = pre_processing(source_x, s_dict, target_x, t_dict, w_embed, batch_size, device)

        u_rev = text_conv(u_rev).transpose(1, 3)
        i_rev = text_conv(i_rev).transpose(1, 3)
        aux_rev = text_conv(aux_rev).transpose(1, 3)
        # 64, 1, 496, 32

        u_rev = agc(u_rev).transpose(2, 3)
        i_rev = agc(i_rev).transpose(2, 3)
        aux_rev = agc(aux_rev).transpose(2, 3)
        # 64, 1, 16, 496

        # Source
        s_u_rev = s_aat(u_rev).squeeze()
        s_aux_rev = s_aat(aux_rev).squeeze()
        # 64, 16, 16

        s_i_rev = s_aat(i_rev).squeeze()
        s_u_rev = (s_u_rev + s_aux_rev) / 2

        s_u_rev = s_u_rev.reshape(64, 256, 1).squeeze()
        s_i_rev = s_i_rev.reshape(64, 256, 1).squeeze()

        out = torch.cat((s_u_rev, s_i_rev), 1)

        out = s_cls(out)

        masking = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).view(batch_size * 2, -1).to(device)
        out, s_label = torch.mul(out, masking), torch.mul(label, masking)

        s_loss = criterion(out, s_label) * 2

        # Target
        t_u_rev = t_aat(u_rev).squeeze()
        t_aux_rev = t_aat(aux_rev).squeeze()
        # 64, 16, 16

        t_i_rev = t_aat(i_rev).squeeze()
        t_u_rev = (t_u_rev + t_aux_rev) / 2

        t_u_rev = t_u_rev.reshape(64, 256, 1).squeeze()
        t_i_rev = t_i_rev.reshape(64, 256, 1).squeeze()

        out = torch.cat((t_u_rev, t_i_rev), 1)

        out = t_cls(out)

        masking = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)]).view(batch_size * 2, -1).to(device)
        out, t_label = torch.mul(out, masking), torch.mul(label, masking)

        t_loss = criterion(out, t_label) * 2

        loss = s_loss + t_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        print('Source Loss: %.2f, Target Loss: %.2f' % (s_loss, t_loss))

        torch.save(model.state_dict(), save)
