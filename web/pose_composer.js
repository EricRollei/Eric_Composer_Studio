/**
 * pose_composer.js  v14 - Eric_Composer_Studio
 *
 * Fixed BODY_LIMBS order to match official controlnet_aux limbSeq exactly.
 * Neck→nose is now at index 12 (was at index 0), fixing color assignments.
 */

import { app } from "../../scripts/app.js";

const NODE_TYPE    = "EricPoseComposer";
const W_HASH       = "content_hash";
const W_DETECTED   = "detected_poses_json";
const W_COMP       = "composition_data";

const PANEL_H      = 560;
const PERSON_H     = 24;
const MIN_PERS_PNL = 56;
const RATIO_BAR_H  = 34;
const MARGIN       = 10;
const MIN_COMP     = 80;
const PIXEL_SCALE  = 1024 / 440;
const MOV_R        = 9;
const SCALE_R      = 8;
const CORNER_R     = 14;
const LG_RESIZE    = 16;
const ROT_R        = 7;
const SKEW_R       = 7;
const ROT_DIST     = 40;
const MAX_LIMB_FRAC = 0.55;

const EYE_X   = 13;
const BADGE_X = 40;
const SCALE_X = 65;
const BADGE_W = 22;
const BADGE_H = 14;

const RATIOS = [
    {label:"3:1",  w:3,  h:1 }, {label:"2:1",  w:2,  h:1 },
    {label:"16:9", w:16, h:9 }, {label:"3:2",  w:3,  h:2 },
    {label:"7:5",  w:7,  h:5 }, {label:"7:6",  w:7,  h:6 },
    {label:"5:4",  w:5,  h:4 }, {label:"1:1",  w:1,  h:1 },
    {label:"4:5",  w:4,  h:5 }, {label:"6:7",  w:6,  h:7 },
    {label:"5:7",  w:5,  h:7 }, {label:"2:3",  w:2,  h:3 },
    {label:"9:16", w:9,  h:16}, {label:"1:2",  w:1,  h:2 },
    {label:"1:3",  w:1,  h:3 },
];

function findClosestRatio(cw, ch) {
    if (!ch) return RATIOS[7];
    const aspect = cw / ch;
    let best = RATIOS[0], bestDiff = Infinity;
    for (const r of RATIOS) {
        const diff = Math.abs(r.w / r.h - aspect);
        if (diff < bestDiff) { best = r; bestDiff = diff; }
    }
    return best;
}
function snapToRatio(node, ratioW, ratioH) {
    const comp = getComp(node) || {};
    const cw   = comp.display_w || 440;
    setDispDims(node, cw, Math.max(MIN_COMP, Math.round(cw * ratioH / ratioW)));
}

// BODY_LIMBS order matches controlnet_aux util.py limbSeq exactly (0-indexed).
// neck→nose is at index 12 (blue), NOT index 0 - putting it at 0 shifts all colors.
const BODY_LIMBS = [
    [1,2],[1,5],[2,3],[3,4],[5,6],[6,7],        // shoulders/arms
    [1,8],[8,9],[9,10],[1,11],[11,12],[12,13],  // hips/legs
    [1,0],[0,14],[14,16],[0,15],[15,17],         // neck-face
    [8,11],                                      // pelvis (our extra)
];
const HAND_LIMBS = [
    [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],
];
const FOOT_LIMBS = [[2,0],[2,1],[5,3],[5,4]];

const _gw  = (n,k)   => n.widgets?.find(w => w.name === k);
const gval = (n,k,d) => _gw(n,k)?.value ?? d;
const sval = (n,k,v) => { const w = _gw(n,k); if(w) w.value = v; };
const getDetected = n => { try{ return JSON.parse(gval(n,W_DETECTED,"")||"null"); }catch{return null;} };
const getComp     = n => { try{ return JSON.parse(gval(n,W_COMP,    "")||"null"); }catch{return null;} };
const setComp     = (n,d) => sval(n, W_COMP, JSON.stringify(d));
const c3hex = ([r,g,b]) => "#"+[r,g,b].map(x=>x.toString(16).padStart(2,"0")).join("");

function getDispDims(node, panelInnerW) {
    const comp = getComp(node);
    const pH   = node._panelH || PANEL_H;
    const maxW = Math.min(panelInnerW - 4, 560);
    return {
        w: Math.min(Math.max(comp?.display_w || maxW, MIN_COMP), panelInnerW - 4),
        h: Math.min(Math.max(comp?.display_h || maxW, MIN_COMP), pH - 2*MARGIN - 4),
    };
}
function setDispDims(node, dw, dh) {
    const comp = getComp(node) || {};
    comp.display_w     = Math.round(dw);
    comp.display_h     = Math.round(dh);
    comp.canvas_width  = Math.round(dw * PIXEL_SCALE / 8) * 8;
    comp.canvas_height = Math.round(dh * PIXEL_SCALE / 8) * 8;
    setComp(node, comp);
}
function getOutDims(node) {
    const comp = getComp(node);
    return [comp?.canvas_width || 1024, comp?.canvas_height || 1024];
}

function centNorm(person, srcW, srcH) {
    const flat = person.pose_keypoints_2d ?? [];
    const xs=[], ys=[];
    for(let i=0; i<Math.min(flat.length,54); i+=3)
        if(flat[i+2]>0){ xs.push(flat[i]/srcW); ys.push(flat[i+1]/srcH); }
    return xs.length ? [xs.reduce((a,b)=>a+b)/xs.length, ys.reduce((a,b)=>a+b)/ys.length] : [0.5,0.5];
}
function centSc(xf, ca) { return [ca.x + xf.x * ca.w, ca.y + xf.y * ca.h]; }

function kpSc(kpX, kpY, srcW, srcH, cx_n, cy_n, xf, ca) {
    const [cx_s, cy_s] = centSc(xf, ca);
    const sx  = xf.scale_x  ?? 1.0;
    const sy  = xf.scale_y  ?? 1.0;
    const rot = xf.rotation ?? 0.0;
    const dx  = (kpX/srcW - cx_n) * xf.scale * sx * ca.w;
    const dy  = (kpY/srcH - cy_n) * xf.scale * sy * ca.w;
    const cos_a = Math.cos(rot), sin_a = Math.sin(rot);
    return [cx_s + dx*cos_a - dy*sin_a, cy_s + dx*sin_a + dy*cos_a];
}

function bboxSc(person, srcW, srcH, cx_n, cy_n, xf, ca) {
    const flat = person.pose_keypoints_2d ?? [];
    let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
    for(let i=0; i<Math.min(flat.length,54); i+=3){
        if(flat[i+2]<=0) continue;
        const [sx,sy]=kpSc(flat[i],flat[i+1],srcW,srcH,cx_n,cy_n,xf,ca);
        x0=Math.min(x0,sx);y0=Math.min(y0,sy);x1=Math.max(x1,sx);y1=Math.max(y1,sy);
    }
    return isFinite(x0) ? {x0,y0,x1,y1} : null;
}

const flatKps = (flat,n) => { const o=[]; for(let i=0;i<n*3;i+=3) o.push({x:flat[i]??0,y:flat[i+1]??0,c:flat[i+2]??0}); return o; };

function drawSkeleton(ctx, person, sFn, color, lineW, dotR, alpha, maxLimbPx) {
    ctx.save(); ctx.globalAlpha=alpha;
    const body=flatKps(person.pose_keypoints_2d??[],18);
    const lh  =flatKps(person.hand_left_keypoints_2d??[],21);
    const rh  =flatKps(person.hand_right_keypoints_2d??[],21);
    const ft  =flatKps(person.foot_keypoints_2d??[],6);
    const handMax = maxLimbPx * 0.25;
    const limbOk  = (ax,ay,bx,by,mx) => Math.hypot(bx-ax,by-ay) <= mx;
    const limb=(kps,i,j,c,mx)=>{
        const a=kps[i],b=kps[j]; if(!a||!b||a.c<=0||b.c<=0) return;
        const [ax,ay]=sFn(a.x,a.y),[bx,by]=sFn(b.x,b.y);
        if(!limbOk(ax,ay,bx,by,mx)) return;
        ctx.strokeStyle=c; ctx.lineWidth=lineW;
        ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);ctx.stroke();
    };
    const dot=(kp,c,r)=>{ if(!kp||kp.c<=0) return; const [sx,sy]=sFn(kp.x,kp.y); ctx.fillStyle=c;ctx.beginPath();ctx.arc(sx,sy,r,0,Math.PI*2);ctx.fill(); };
    BODY_LIMBS.forEach(([i,j])=>{ if(i<body.length&&j<body.length) limb(body,i,j,color,maxLimbPx); });
    body.forEach(kp=>dot(kp,color,dotR));
    const hc=color+"88";
    HAND_LIMBS.forEach(([i,j])=>{ if(i<lh.length&&j<lh.length) limb(lh,i,j,hc,handMax); if(i<rh.length&&j<rh.length) limb(rh,i,j,hc,handMax); });
    FOOT_LIMBS.forEach(([i,j])=>{ if(i<ft.length&&j<ft.length) limb(ft,i,j,color+"77",maxLimbPx); });
    ctx.restore();
}

function drawHandle(ctx, x, y, r, fillColor, strokeColor, label, round=false) {
    ctx.save();
    ctx.fillStyle=fillColor; ctx.strokeStyle=strokeColor; ctx.lineWidth=1.5;
    if (round) {
        ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill(); ctx.stroke();
    } else {
        ctx.fillRect(x-r,y-r,r*2,r*2); ctx.strokeRect(x-r,y-r,r*2,r*2);
    }
    if (label) {
        ctx.fillStyle="#FFF"; ctx.font=`bold ${r+3}px sans-serif`;
        ctx.textAlign="center"; ctx.textBaseline="middle";
        ctx.fillText(label, x, y+1);
    }
    ctx.restore();
}

function drawModelBadge(ctx, cx, cy, model) {
    const isDW = (model === "dwpose");
    const bx   = cx - BADGE_W/2, by = cy - BADGE_H/2;
    ctx.save();
    ctx.fillStyle   = isDW ? "#0A1F0A" : "#0A0A1F";
    ctx.strokeStyle = isDW ? "#44AA44" : "#4466CC";
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.roundRect(bx, by, BADGE_W, BADGE_H, 3);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle    = isDW ? "#66DD66" : "#6688EE";
    ctx.font         = "bold 8px monospace";
    ctx.textAlign    = "center"; ctx.textBaseline = "middle";
    ctx.fillText(isDW ? "DW" : "RM", cx, cy);
    ctx.restore();
}

function drawAll(ctx, node, W) {
    const det  = getDetected(node);
    const comp = getComp(node);
    const st   = node._cst;
    const [tW,tH] = getOutDims(node);
    const panelInnerW = W - 2*MARGIN;
    const dims = getDispDims(node, panelInnerW);
    const cw = dims.w, ch = dims.h;
    const ca = { x: MARGIN, y: MARGIN, w: cw, h: ch };
    const maxLimbPx = Math.max(cw, ch) * MAX_LIMB_FRAC;

    const pH = node._panelH || PANEL_H;
    ctx.fillStyle="#06060F"; ctx.fillRect(0,0,W,pH);
    ctx.strokeStyle="#1E2A38"; ctx.lineWidth=1; ctx.strokeRect(1,1,W-2,pH-2);
    ctx.fillStyle="#080810"; ctx.fillRect(ca.x,ca.y,ca.w,ca.h);
    ctx.strokeStyle="#334455"; ctx.lineWidth=1; ctx.strokeRect(ca.x,ca.y,ca.w,ca.h);

    ctx.save(); ctx.strokeStyle="#141E2A"; ctx.lineWidth=0.5;
    for(let i=1;i<3;i++){
        const gx=ca.x+ca.w*i/3, gy=ca.y+ca.h*i/3;
        ctx.beginPath();ctx.moveTo(gx,ca.y);ctx.lineTo(gx,ca.y+ca.h);ctx.stroke();
        ctx.beginPath();ctx.moveTo(ca.x,gy);ctx.lineTo(ca.x+ca.w,gy);ctx.stroke();
    }
    ctx.restore();

    ctx.fillStyle="#2A3A4A"; ctx.font="10px monospace"; ctx.textAlign="center"; ctx.textBaseline="bottom";
    ctx.fillText(`${tW} × ${tH}`, ca.x+ca.w/2, ca.y+ca.h-2);

    if(!det?.persons?.length){
        ctx.fillStyle="#445566"; ctx.font="13px sans-serif"; ctx.textBaseline="middle"; ctx.textAlign="center";
        ctx.fillText("Connect inputs and queue to detect poses", ca.x+ca.w/2, ca.y+ca.h/2);
    } else {
        st._bboxes={};st._handles={};
        det.persons.forEach(info=>{
            const xf=comp?.persons?.find(p=>p.id===info.id);
            if(xf?.visible===false) return;
            const [cx_n,cy_n]=centNorm(info.person,info.src_w,info.src_h);
            const tfm=xf??{x:cx_n,y:cy_n,scale:1.0,scale_x:1.0,scale_y:1.0,rotation:0.0,visible:true};
            const col=c3hex(info.color);
            const sel=(st?.selId===info.id);
            const sFn=(kx,ky)=>kpSc(kx,ky,info.src_w,info.src_h,cx_n,cy_n,tfm,ca);
            drawSkeleton(ctx,info.person,sFn,col,sel?2.5:1.8,sel?4.5:3,sel?1.0:0.7,maxLimbPx);
            const [csx,csy]=centSc(tfm,ca);
            ctx.save(); ctx.strokeStyle=sel?"#FFF":col; ctx.fillStyle=sel?"#FFF3":col+"33"; ctx.lineWidth=sel?2:1;
            ctx.beginPath();ctx.arc(csx,csy,MOV_R,0,Math.PI*2);ctx.fill();ctx.stroke(); ctx.restore();
            if(sel){
                const bb=bboxSc(info.person,info.src_w,info.src_h,cx_n,cy_n,tfm,ca);
                if(bb){
                    st._bboxes[info.id]=bb;
                    const bcx=(bb.x0+bb.x1)/2;
                    ctx.save(); ctx.fillStyle="#1A2A3A"; ctx.strokeStyle="#88AAFF"; ctx.lineWidth=1.5;
                    ctx.fillRect(bb.x1-SCALE_R,bb.y0-SCALE_R,SCALE_R*2,SCALE_R*2);
                    ctx.strokeRect(bb.x1-SCALE_R,bb.y0-SCALE_R,SCALE_R*2,SCALE_R*2);
                    ctx.strokeStyle="#5588CC"; ctx.lineWidth=1; ctx.beginPath();
                    ctx.moveTo(bb.x1-SCALE_R+2,bb.y0+SCALE_R-2); ctx.lineTo(bb.x1+SCALE_R-2,bb.y0-SCALE_R+2); ctx.stroke(); ctx.restore();
                    const rhx=bcx, rhy=bb.y0-ROT_DIST;
                    ctx.save(); ctx.strokeStyle="#554400"; ctx.lineWidth=1; ctx.setLineDash([3,2]);
                    ctx.beginPath(); ctx.moveTo(bcx,bb.y0); ctx.lineTo(rhx,rhy); ctx.stroke();
                    ctx.setLineDash([]); ctx.restore();
                    drawHandle(ctx,rhx,rhy,ROT_R,"#332200","#FFAA22","↻",true);
                    const sxhx=bb.x1+SKEW_R+6, sxhy=(bb.y0+bb.y1)/2;
                    drawHandle(ctx,sxhx,sxhy,SKEW_R,"#0A1A2A","#4488CC","↔");
                    const syhx=bcx, syhy=bb.y1+SKEW_R+6;
                    drawHandle(ctx,syhx,syhy,SKEW_R,"#0A1F0A","#44AA44","↕");
                    st._handles[info.id]={rot:{x:rhx,y:rhy},scaleX:{x:sxhx,y:sxhy},scaleY:{x:syhx,y:syhy}};
                    const rot_deg=Math.round((tfm.rotation??0)*180/Math.PI);
                    const sxv=(tfm.scale_x??1).toFixed(2), syv=(tfm.scale_y??1).toFixed(2);
                    if(rot_deg!==0||sxv!=="1.00"||syv!=="1.00"){
                        ctx.fillStyle="#334455"; ctx.font="9px monospace";
                        ctx.textAlign="center"; ctx.textBaseline="top";
                        ctx.fillText(`${rot_deg}°  sx${sxv}  sy${syv}`,csx,csy+MOV_R+3);
                    }
                }
            }
        });
    }

    const rx=ca.x+ca.w, ry=ca.y+ca.h;
    const isResizing=(st?.dragMode==="resize");
    ctx.save(); ctx.fillStyle=isResizing?"#5599FFCC":"#33445588";
    ctx.beginPath(); ctx.moveTo(rx-CORNER_R*2.5,ry); ctx.lineTo(rx,ry-CORNER_R*2.5); ctx.lineTo(rx,ry); ctx.closePath(); ctx.fill();
    ctx.fillStyle=isResizing?"#88BBFF":"#556677"; ctx.beginPath(); ctx.arc(rx,ry,4,0,Math.PI*2); ctx.fill(); ctx.restore();
    const closestRatio=findClosestRatio(cw,ch);
    ctx.save(); ctx.font=isResizing?"bold 11px monospace":"11px monospace";
    ctx.fillStyle=isResizing?"#88CCFF":"#446688"; ctx.textAlign="left"; ctx.textBaseline="top";
    ctx.fillText(`≈ ${closestRatio.label}`,rx+4,ry-24);
    ctx.fillStyle="#334455"; ctx.font="9px monospace";
    ctx.fillText(`${Math.round(cw)}×${Math.round(ch)}px`,rx+4,ry-10); ctx.restore();

    const listY=pH+2;
    const n=det?.persons?.length??0;
    const panelH2=Math.max(n*PERSON_H+20,MIN_PERS_PNL);
    ctx.fillStyle="#0C0C1A"; ctx.fillRect(0,listY,W,panelH2);
    ctx.strokeStyle="#1E2A38"; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(0,listY); ctx.lineTo(W,listY); ctx.stroke();
    if(!n){
        ctx.fillStyle="#2A3A4A"; ctx.font="11px sans-serif"; ctx.textAlign="center"; ctx.textBaseline="middle";
        ctx.fillText("No persons detected yet",W/2,listY+MIN_PERS_PNL/2); return;
    }
    ctx.font="11px sans-serif"; ctx.textBaseline="middle";
    det.persons.forEach((info,i)=>{
        const y=listY+6+i*PERSON_H;
        const xf=comp?.persons?.find(p=>p.id===info.id);
        const vis=xf?.visible!==false, sel=(st?.selId===info.id), col=c3hex(info.color);
        const isPhotoSource=(info.photo_idx!=null && info.det_model!=="external");
        ctx.fillStyle=sel?"#1E3048":"#11111F"; ctx.fillRect(MARGIN,y,W-2*MARGIN,PERSON_H-2);
        ctx.fillStyle=col; ctx.fillRect(MARGIN+4,y+5,10,PERSON_H-10);
        ctx.fillStyle=sel?"#7ABAEE":"#4A7A9A"; ctx.textAlign="right";
        ctx.fillText(`×${(xf?.scale??1.0).toFixed(2)}`, W-MARGIN-SCALE_X, y+PERSON_H/2);
        if(isPhotoSource){
            const photoModels=comp?.photo_models??{};
            const model=photoModels[String(info.photo_idx)]??info.det_model??"rtmw";
            drawModelBadge(ctx, W-MARGIN-BADGE_X, y+PERSON_H/2, model);
        }
        ctx.fillStyle=vis?"#CCDDE8":"#445566"; ctx.textAlign="left";
        ctx.fillText(info.label, MARGIN+20, y+PERSON_H/2);
        const ex=W-MARGIN-EYE_X, ey=y+PERSON_H/2;
        ctx.strokeStyle=vis?"#5588AA":"#2A3A4A"; ctx.lineWidth=1.5;
        ctx.beginPath();ctx.ellipse(ex,ey,7,4,0,0,Math.PI*2);ctx.stroke();
        if(vis){ ctx.fillStyle="#5588AA"; ctx.beginPath();ctx.arc(ex,ey,2,0,Math.PI*2);ctx.fill(); }
    });
}

function setup(node) {
    const wrapper = document.createElement("div");
    wrapper.style.cssText = "width:100%;overflow:hidden;user-select:none;pointer-events:all;display:flex;flex-direction:column;";

    const ratioBar = document.createElement("div");
    ratioBar.style.cssText = ["display:flex","align-items:center","gap:6px",
        "padding:4px 10px","background:#0A0A18","border-bottom:1px solid #1E2A38",
        `height:${RATIO_BAR_H}px`,"box-sizing:border-box","flex-shrink:0"].join(";");

    const ratioLabel = document.createElement("span");
    ratioLabel.textContent = "Snap:";
    ratioLabel.style.cssText = "color:#446688;font:11px monospace;white-space:nowrap;";

    const ratioSelect = document.createElement("select");
    ratioSelect.style.cssText = "background:#12121E;color:#88AACC;border:1px solid #2A3A4A;border-radius:3px;font:11px monospace;padding:2px 4px;cursor:pointer;pointer-events:all;";
    const freeOpt=document.createElement("option"); freeOpt.value=""; freeOpt.textContent="- free -";
    ratioSelect.appendChild(freeOpt);
    RATIOS.forEach(r=>{ const o=document.createElement("option"); o.value=`${r.w}:${r.h}`; o.textContent=r.label; ratioSelect.appendChild(o); });
    ratioSelect.addEventListener("change",()=>{ const v=ratioSelect.value; if(!v) return; const [rw,rh]=v.split(":").map(Number); snapToRatio(node,rw,rh); node._pcRedraw?.(); });

    const modeBtn=document.createElement("button");
    modeBtn.style.cssText="background:#12121E;border:1px solid #2A3A4A;border-radius:3px;font:11px monospace;padding:2px 8px;cursor:pointer;pointer-events:all;white-space:nowrap;";
    function getCurrentMode(){ return getComp(node)?.color_mode||"dwpose"; }
    function syncModeBtn(){
        const m=getCurrentMode();
        modeBtn.textContent=m==="dwpose"?"DWPose":"Enhanced";
        modeBtn.style.color=m==="dwpose"?"#88AACC":"#FFCC66";
        modeBtn.style.borderColor=m==="dwpose"?"#2A3A4A":"#665522";
        modeBtn.style.background=m==="dwpose"?"#12121E":"#1A1408";
    }
    syncModeBtn();
    modeBtn.addEventListener("click",()=>{ const c=getComp(node); if(!c) return; c.color_mode=getCurrentMode()==="dwpose"?"enhanced":"dwpose"; setComp(node,c); syncModeBtn(); });

    const xinsrBtn=document.createElement("button");
    xinsrBtn.title="Scale stick thickness for xinsir/controlnet-openpose-sdxl-1.0. At 1024px: stickwidth=12. DWPose mode only.";
    xinsrBtn.style.cssText="background:#12121E;border:1px solid #2A3A4A;border-radius:3px;font:10px monospace;padding:2px 6px;cursor:pointer;pointer-events:all;white-space:nowrap;";
    function getXinsr(){ return getComp(node)?.xinsr_stick_scaling===true; }
    function syncXinsrBtn(){
        const on=getXinsr();
        xinsrBtn.textContent=on?"Xinsr ✓":"Xinsr";
        xinsrBtn.style.color=on?"#FFAA44":"#446688";
        xinsrBtn.style.borderColor=on?"#664422":"#2A3A4A";
        xinsrBtn.style.background=on?"#1A1000":"#12121E";
    }
    syncXinsrBtn();
    xinsrBtn.addEventListener("click",()=>{ const c=getComp(node); if(!c) return; c.xinsr_stick_scaling=!getXinsr(); setComp(node,c); syncXinsrBtn(); });

    const refreshBtn=document.createElement("button");
    node._refreshBtn=refreshBtn;
    refreshBtn.textContent="⟳ Refresh";
    refreshBtn.style.cssText=["margin-left:auto","background:#0A1A0A","color:#44AA44",
        "border:1px solid #226622","border-radius:3px","font:11px monospace","padding:2px 8px",
        "cursor:pointer","pointer-events:all","white-space:nowrap"].join(";");
    refreshBtn.addEventListener("click",()=>{
        sval(node,W_HASH,""); refreshBtn.textContent="✓ Queued"; refreshBtn.style.color="#88FF88";
        setTimeout(()=>{ refreshBtn.textContent="⟳ Refresh"; refreshBtn.style.color="#44AA44"; },1500);
    });

    ratioBar.appendChild(ratioLabel); ratioBar.appendChild(ratioSelect);
    ratioBar.appendChild(modeBtn); ratioBar.appendChild(xinsrBtn); ratioBar.appendChild(refreshBtn);
    wrapper.appendChild(ratioBar);

    const canvas=document.createElement("canvas");
    canvas.style.cssText="display:block;width:100%;pointer-events:all;touch-action:none;cursor:default;flex-shrink:0;";
    wrapper.appendChild(canvas);

    node._cst={selId:null,dragMode:null,_bboxes:{},_handles:{}};
    node._panelH = PANEL_H;
    const gW=()=>Math.max(canvas.clientWidth||wrapper.clientWidth||600,150);
    function totalH(){ const n=getDetected(node)?.persons?.length??0; return (node._panelH||PANEL_H)+Math.max(n*PERSON_H+20,MIN_PERS_PNL)+16; }

    function redraw(){
        const W=gW(),th=totalH(),dpr=window.devicePixelRatio||1;
        canvas.width=W*dpr; canvas.height=th*dpr; canvas.style.height=th+"px";
        const ctx=canvas.getContext("2d");
        ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,th);
        drawAll(ctx,node,W);
        const comp=getComp(node);
        if(comp?.display_w&&comp?.display_h){ const cl=findClosestRatio(comp.display_w,comp.display_h); ratioSelect.value=`${cl.w}:${cl.h}`; }
        syncModeBtn(); syncXinsrBtn();
    }
    node._pcRedraw=redraw;

    const widget=node.addDOMWidget("_cc","custom",wrapper,{ serialize:false,hideOnZoom:false,getValue:()=>"",setValue:()=>{} });
    widget.computeSize=w=>[w, RATIO_BAR_H+totalH()];
    setTimeout(redraw,200);

    function getPanelInnerW(){ return gW()-2*MARGIN; }
    function getCA(){ const d=getDispDims(node,getPanelInnerW()); return {x:MARGIN,y:MARGIN,w:d.w,h:d.h}; }

    canvas.addEventListener("pointerdown", e=>{
        canvas.setPointerCapture(e.pointerId); e.preventDefault(); e.stopPropagation();
        const x=e.offsetX, y=e.offsetY, W=gW(), ca=getCA();
        const det=getDetected(node), comp=getComp(node), st=node._cst;
        const cx=ca.x+ca.w, cy=ca.y+ca.h;
        if(Math.abs(x-cx)<=CORNER_R+4&&Math.abs(y-cy)<=CORNER_R+4){
            st.dragMode="resize"; st.resStartX=x; st.resStartY=y;
            st.resStartCW=ca.w; st.resStartCH=ca.h;
            st.resMaxW=getPanelInnerW()-4; st.resMaxH=(node._panelH||PANEL_H)-2*MARGIN-4; return;
        }
        const listY=(node._panelH||PANEL_H)+6;
        if(y>=listY&&det?.persons?.length){
            const i=Math.floor((y-listY)/PERSON_H);
            if(i>=0&&i<det.persons.length){
                const info=det.persons[i]; const pid=info.id;
                if(x>=W-MARGIN-EYE_X-10&&x<=W-MARGIN-EYE_X+10&&comp){
                    const p=comp.persons?.find(p=>p.id===pid);
                    if(p){ p.visible=!(p.visible!==false); setComp(node,comp); redraw(); return; }
                }
                const isPhotoSource=(info.photo_idx!=null && info.det_model!=="external");
                if(isPhotoSource){
                    const badgeCX=W-MARGIN-BADGE_X;
                    if(Math.abs(x-badgeCX)<=BADGE_W/2+4){
                        if(!comp.photo_models) comp.photo_models={};
                        const curModel=comp.photo_models[String(info.photo_idx)]??"rtmw";
                        const newModel=curModel==="rtmw"?"dwpose":"rtmw";
                        comp.photo_models[String(info.photo_idx)]=newModel;
                        setComp(node,comp); sval(node,W_HASH,"");
                        const rb=node._refreshBtn;
                        if(rb){ rb.textContent=`⟳ Re-detect (${newModel==="dwpose"?"DW":"RM"})`; rb.style.color="#FFCC44"; setTimeout(()=>{ rb.textContent="⟳ Refresh"; rb.style.color="#44AA44"; },3000); }
                        redraw(); return;
                    }
                }
                st.selId=(st.selId===pid)?null:pid; redraw(); return;
            }
        }
        if(!det?.persons?.length||!comp){ st.selId=null; redraw(); return; }
        if(st.selId && st._handles?.[st.selId]){
            const hn=st._handles[st.selId];
            const xf=comp.persons?.find(p=>p.id===st.selId);
            if(xf){
                const [csx,csy]=centSc(xf,ca);
                if(Math.hypot(x-hn.rot.x,y-hn.rot.y)<=ROT_R+6){
                    st.dragMode="rotate"; st.dragId=st.selId; st.origRotation=xf.rotation??0;
                    st.rotStartAngle=Math.atan2(y-csy,x-csx); st.rotCx=csx; st.rotCy=csy; return;
                }
                if(Math.hypot(x-hn.scaleX.x,y-hn.scaleX.y)<=SKEW_R+6){
                    st.dragMode="scaleX"; st.dragId=st.selId; st.origScaleX=xf.scale_x??1.0;
                    st.skewStartX=x; st.skewCx=csx; st.origDistX=Math.abs(hn.scaleX.x-csx)||1; return;
                }
                if(Math.hypot(x-hn.scaleY.x,y-hn.scaleY.y)<=SKEW_R+6){
                    st.dragMode="scaleY"; st.dragId=st.selId; st.origScaleY=xf.scale_y??1.0;
                    st.skewStartY=y; st.skewCy=csy; st.origDistY=Math.abs(hn.scaleY.y-csy)||1; return;
                }
            }
        }
        if(st.selId&&st._bboxes?.[st.selId]){
            const bb=st._bboxes[st.selId];
            if(Math.abs(x-bb.x1)<=SCALE_R+5&&Math.abs(y-bb.y0)<=SCALE_R+5){
                const xf=comp.persons?.find(p=>p.id===st.selId);
                if(xf){ const [ax,ay]=centSc(xf,ca); st.dragMode="scale"; st.dragId=st.selId; st.origScale=xf.scale; st.origDist=Math.hypot(bb.x1-ax,bb.y0-ay)||1; st.anchorSx=ax; st.anchorSy=ay; return; }
            }
        }
        if(x>=ca.x&&x<=ca.x+ca.w&&y>=ca.y&&y<=ca.y+ca.h){
            let best=null, bestD=MOV_R+12;
            (det.persons||[]).forEach(info=>{ const xf=comp.persons?.find(p=>p.id===info.id); if(!xf) return; const [sx,sy]=centSc(xf,ca); const d=Math.hypot(x-sx,y-sy); if(d<bestD){ best=info; bestD=d; } });
            if(best){ st.selId=best.id; st.dragMode="move"; st.dragId=best.id; st.startX=x; st.startY=y; const xf=comp.persons.find(p=>p.id===best.id); st.origX=xf.x; st.origY=xf.y; }
            else st.selId=null;
        } else st.selId=null;
        redraw();
    },{passive:false});

    canvas.addEventListener("pointermove", e=>{
        const st=node._cst; if(!st?.dragMode) return;
        e.preventDefault(); e.stopPropagation();
        const x=e.offsetX, y=e.offsetY, ca=getCA(), comp=getComp(node);
        if(st.dragMode==="resize"){
            const newCW=Math.max(MIN_COMP,Math.min(st.resStartCW+(x-st.resStartX),st.resMaxW));
            const newCH=Math.max(MIN_COMP,Math.min(st.resStartCH+(y-st.resStartY),st.resMaxH));
            setDispDims(node,newCW,newCH); requestAnimationFrame(redraw); return;
        }
        if(!comp) return;
        const xf=comp.persons?.find(p=>p.id===st.dragId); if(!xf) return;
        if(st.dragMode==="move"){
            xf.x=st.origX+(x-st.startX)/ca.w; xf.y=st.origY+(y-st.startY)/ca.h;
        } else if(st.dragMode==="scale"){
            const d=Math.hypot(x-st.anchorSx,y-st.anchorSy);
            xf.scale=Math.max(0.05,Math.min(5.0,st.origScale*d/st.origDist));
        } else if(st.dragMode==="rotate"){
            xf.rotation=st.origRotation+(Math.atan2(y-st.rotCy,x-st.rotCx)-st.rotStartAngle);
        } else if(st.dragMode==="scaleX"){
            xf.scale_x=Math.max(0.05,Math.min(5.0,st.origScaleX*Math.abs(x-st.skewCx)/st.origDistX));
        } else if(st.dragMode==="scaleY"){
            xf.scale_y=Math.max(0.05,Math.min(5.0,st.origScaleY*Math.abs(y-st.skewCy)/st.origDistY));
        }
        setComp(node,comp); requestAnimationFrame(redraw);
    },{passive:false});

    canvas.addEventListener("pointerup", e=>{ if(node._cst) node._cst.dragMode=null; try{ canvas.releasePointerCapture(e.pointerId); }catch{} redraw(); });

    const origMD=node.onMouseDown?.bind(node);
    node.onMouseDown=function(e,pos,graph){
        if(pos[1]<0) return origMD?.(e,pos,graph);
        if(pos[0]>node.size[0]-LG_RESIZE&&pos[1]>node.size[1]-LG_RESIZE) return origMD?.(e,pos,graph);
        return true;
    };

    function tryHide(n){
        for(const k of [W_HASH,W_DETECTED,W_COMP]){ const w=_gw(node,k); if(w){ w.computeSize=()=>[0,-4]; w.draw=()=>{}; w.type="converted-widget"; } }
        if(n<12) setTimeout(()=>tryHide(n+1),200);
        else { app.graph.setDirtyCanvas(true,true); setTimeout(redraw,50); }
    }
    setTimeout(()=>tryHide(0),80);

    const origEx=node.onExecuted?.bind(node);
    node.onExecuted=function(msg){
        origEx?.(msg);
        const d=msg?.pose_composer_data?.[0]; if(!d) return;
        if(d.content_hash)        sval(this,W_HASH,    d.content_hash);
        if(d.detected_poses_json) sval(this,W_DETECTED,d.detected_poses_json);
        if(d.composition_data)    sval(this,W_COMP,    d.composition_data);
        setTimeout(()=>this._pcRedraw?.(),60);
    };

    return widget;
}

app.registerExtension({
    name:"Eric.PoseComposer",
    async nodeCreated(node){
        if(node.comfyClass!==NODE_TYPE) return;
        if(!node.size||node.size[0]<500) node.size=[760,800];
        setup(node);
    },
});
