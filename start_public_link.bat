@echo off
chcp 65001 >nul
echo ============================================================================
echo   GrassSense x Pasture.io — ระบบเปิดลิงก์ออนไลน์สาธารณะให้เพื่อนเข้าใช้งาน
echo ============================================================================
echo.
echo [1/2] กำลังรันเซิร์ฟเวอร์หลังบ้านพอร์ต 8000...
start "GrassSense Server" cmd /k "python -m uvicorn app:app --host 0.0.0.0 --port 8000"
timeout /t 3 >nul
echo [2/2] กำลังสร้างลิงก์ HTTPS สาธารณะผ่าน Cloudflare Tunnel...
echo เมื่อลิงก์ปรากฏด้านล่างในกรอบสี่เหลี่ยม (https://....trycloudflare.com)
echo ให้คัดลอกลิงก์นั้นส่งให้เพื่อนเปิดผ่านมือถือหรือคอมพิวเตอร์ได้ทันที!
echo.
.\cloudflared.exe tunnel --url http://127.0.0.1:8000 --no-autoupdate
pause

