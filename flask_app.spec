# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['flask_app.py'],
             pathex=['/home/cheddartot/Documents/Github/Capstone/Ag-AI'],
             binaries=[],
             datas=[('utilities', 'utilities'), ('app', 'app')],
             hiddenimports=['packaging.requirements', 'pkg_resources.py2_warn'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='flask_app',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='flask_app')
