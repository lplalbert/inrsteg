import sys
with open('inrmark_api.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('sys.path.append("/home/light_sun/workspace/inrmark_2/inrsteg-final_v1")', 'sys.path.append(".")')
text = text.replace('"/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/v6_30bits.yaml"', '"./config/v6_30bits.yaml"')
text = text.replace('"/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/ismark_v6_30bits/lightning_logs/version_0/checkpoints/ckpt-epoch=109-val_loss=0.0481.ckpt"', '"./output/ismark_v6_30bits/lightning_logs/version_0/checkpoints/ckpt-epoch=109-val_loss=0.0481.ckpt"')
text = text.replace('"/home/light_sun/workspace/inrsteg/data/DIV2K_valid"', '"./output/div2k_imgs"')
text = text.replace('"/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/watermark_template/{}.pt"', '"./output/watermark_template/{}.pt"')
text = text.replace('"/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/watermarked_img.png"', '"./output/watermarked_img.png"')

# Make sure cuda:2 won't OOM or conflict with another model running, we can just use cuda:0 which is free
text = text.replace('device="cuda:2"', 'device="cuda:0"')

with open('inrmark_api.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Paths patched.")
