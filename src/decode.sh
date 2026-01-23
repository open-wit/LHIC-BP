CUDA_VISIBLE_DEVICES=7 python -m run_lhic_bp.decode_lhic \
        --config '/home/huangqizhi/codes/python/s2clic_bps/lhic_bp/src/ckpt/best/config.yml' \
        --msp_ckpt_dir '/home/huangqizhi/codes/python/s2clic_bps/lhic_bp/src/ckpt/lhic_bp/msp_ckpt.pth' \
        --lsp_ckpt_dir '/home/huangqizhi/codes/python/s2clic_bps/lhic_bp/src/ckpt/lhic_bp/lsp_ckpt.pth' \
        --bin  /home/huangqizhi/codes/python/s2clic_bps/lhic_bp/src/tmp//encoded_patches.bin\
        --out ./tmp/decoded_patches.npy \
        --data /home/huangqizhi/data/hsi/hyspecnet-11k/patches/ENMAP01-____L2A-DT0000004950_20221103T162438Z_001_V010110_20221118T145147Z/ENMAP01-____L2A-DT0000004950_20221103T162438Z_001_V010110_20221118T145147Z-Y01460273_X03110438/ENMAP01-____L2A-DT0000004950_20221103T162438Z_001_V010110_20221118T145147Z-Y01460273_X03110438-DATA.npy