import torch
import os
from tqdm import tqdm

def get_3d_corners_torch(h, w, l, x, y, z, ry, device):
    h, w, l, x, y, ry = [torch.tensor(v, device=device, dtype=torch.float32) for v in [h, w, l, x, y, ry]]
    zero, one = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
    x_corners = torch.stack([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_corners = torch.stack([zero, zero, zero, zero, -h, -h, -h, -h])
    z_corners = torch.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
    corners_3d = torch.stack([x_corners, y_corners, z_corners])
    cos_ry, sin_ry = torch.cos(ry), torch.sin(ry)
    R = torch.stack([torch.stack([cos_ry, zero, sin_ry]), torch.stack([zero, one, zero]), torch.stack([-sin_ry, zero, cos_ry])])
    corners_3d = torch.matmul(R, corners_3d)
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z 
    return corners_3d

def purify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # [수정] 안내 메시지 변경
    print(f"[*] Final Purification: P1 Reference | Constraint: +/- 5px")
    
    attack_calib_dir = r"your path"
    guide_2d_dir = r"your path"
    infer_3d_dir = r"your path"
    output_dir = r"your path"
    
    os.makedirs(output_dir, exist_ok=True)
    file_list = [f for f in os.listdir(attack_calib_dir) if f.endswith('.txt')]

    for file_name in tqdm(file_list, desc="Precision Matching"):
        with open(os.path.join(attack_calib_dir, file_name), 'r') as f:
            lines = f.readlines()
        
        p1_idx = [i for i, l in enumerate(lines) if l.startswith('P1:')][0]
        p2_idx = [i for i, l in enumerate(lines) if l.startswith('P2:')][0]
        
        p1_v = [float(x) for x in lines[p1_idx].split()[1:]]
        fx_ref, fy_ref, cx_ref, cy_ref = p1_v[0], p1_v[5], p1_v[2], p1_v[6]
        
        p2_v = [float(x) for x in lines[p2_idx].split()[1:]]
        fx = torch.tensor(p2_v[0], device=device, requires_grad=True)
        fy = torch.tensor(p2_v[5], device=device, requires_grad=True)
        cx = torch.tensor(p2_v[2], device=device, requires_grad=True)
        cy = torch.tensor(p2_v[6], device=device, requires_grad=True)

        infer_path = os.path.join(infer_3d_dir, file_name)
        guide_path = os.path.join(guide_2d_dir, file_name)
        
        if os.path.exists(infer_path) and os.path.exists(guide_path):
            with open(infer_path, 'r') as f_i:
                d = f_i.readline().split()
                if d:
                    h, w, l, tx, ty, tz_val, ry = map(float, [d[8], d[9], d[10], d[11], d[12], d[13], d[14]])
                    with open(guide_path, 'r') as f_g:
                        g_l = f_g.readline()
                        if g_l:
                            target_box = torch.tensor([float(x) for x in g_l.split()[-4:]], device=device)
                            curr_tz = torch.tensor(tz_val, device=device, requires_grad=True)
                            optimizer = torch.optim.Adam([fx, fy, cx, cy, curr_tz], lr=0.1)

                            for _ in range(30):
                                optimizer.zero_grad()
                                pts_3d = get_3d_corners_torch(h, w, l, tx, ty, curr_tz, ry, device)
                                row1 = torch.stack([fx, torch.tensor(0.0, device=device), cx, torch.tensor(p2_v[3], device=device)])
                                row2 = torch.stack([torch.tensor(0.0, device=device), fy, cy, torch.tensor(p2_v[7], device=device)])
                                row3 = torch.tensor([0., 0., 1., p2_v[11]], device=device)
                                P = torch.stack([row1, row2, row3])
                                pts_2d = torch.matmul(P, torch.vstack([pts_3d, torch.ones((1, 8), device=device)]))
                                pts_2d = pts_2d[:2] / (pts_2d[2] + 1e-6)
                                proj_box = torch.stack([pts_2d[0].min(), pts_2d[1].min(), pts_2d[0].max(), pts_2d[1].max()])
                                torch.nn.functional.smooth_l1_loss(proj_box, target_box).backward()
                                optimizer.step()

                                # [수정] 업데이트 직후 매번 P1 +/- 5px 범위로 강제 고정
                                with torch.no_grad():
                                    fx.copy_(torch.clamp(fx, fx_ref - 5.0, fx_ref + 5.0))
                                    fy.copy_(torch.clamp(fy, fy_ref - 5.0, fy_ref + 5.0))
                                    cx.copy_(torch.clamp(cx, cx_ref - 5.0, cx_ref + 5.0))
                                    cy.copy_(torch.clamp(cy, cy_ref - 5.0, cy_ref + 5.0))

        # [수정] 모든 숫자를 P1 기준 +/- 5px 내에서 12자리 지수 표기로 통일
        with torch.no_grad():
            f_x, f_y = torch.clamp(fx, fx_ref - 5.0, fx_ref + 5.0).item(), torch.clamp(fy, fy_ref - 5.0, fy_ref + 5.0).item()
            c_x, c_y = torch.clamp(cx, cx_ref - 5.0, cx_ref + 5.0).item(), torch.clamp(cy, cy_ref - 5.0, cy_ref + 5.0).item()

        p2_purified = [
            f_x, 0.0, c_x, p2_v[3],
            0.0, f_y, c_y, p2_v[7],
            0.0, 0.0, 1.0, p2_v[11]
        ]
        
        new_p2_str = "P2: " + " ".join([f"{val:.12e}" for val in p2_purified]) + "\n"
        lines[p2_idx] = new_p2_str
        
        with open(os.path.join(output_dir, file_name), 'w') as f:
            f.writelines(lines)

if __name__ == "__main__":
    purify()