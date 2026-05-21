export const blackHoleFragmentShader = /* glsl */ `
precision highp float;

varying vec2 vUv;

uniform vec2  uResolution;
uniform float uTime;
uniform vec3  uCamPos;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform vec3  uCamForward;
uniform float uFov;
uniform float uLensStrength;
uniform float uDiskPower;
uniform float uExposure;
uniform float uSpinWarp;
uniform float uDiskTilt;
uniform float uJetPower;
uniform float uPhotonEcho;
uniform float uShadowPower;
uniform float uUltra;
uniform float uQualitySteps;
uniform float uPresetMood;
uniform float uFocusMode;
uniform float uSplitMode;
uniform float uMatterPulse;
uniform float uObservatoryMode;
uniform float uDiskHaze;
uniform float uLensDirt;
uniform float uSecondaryImage;
uniform float uTraverseMode;
uniform float uCloudDensity;

#define MAX_STEPS 220
#define FAR_DIST  30.0

const float HORIZON  = 0.86;
const float PHOTON_R = 1.30;
const float DISK_IN  = 1.10;
const float DISK_OUT = 6.0;
const float PI  = 3.14159265359;
const float TAU = 6.28318530718;

// ════════════════════════════════════════════════════════
//  HASH / NOISE
// ════════════════════════════════════════════════════════
float sat(float x){ return clamp(x,0.0,1.0); }
vec3 desat(vec3 c, float a){ float l=dot(c, vec3(0.299,0.587,0.114)); return mix(c, vec3(l), a); }
float h11(float p){ p=fract(p*0.1031); p*=p+33.33; p*=p+p; return fract(p); }
float h21(vec2 p){ vec3 q=fract(vec3(p.xyx)*0.1031); q+=dot(q,q.yzx+33.33); return fract((q.x+q.y)*q.z); }
float h13(vec3 p){ p=fract(p*vec3(0.1031,0.1030,0.0973)); p+=dot(p,p.yxz+33.33); return fract((p.x+p.y)*p.z); }

float n2(vec2 p){
  vec2 i=floor(p),f=fract(p); f=f*f*(3.0-2.0*f);
  return mix(mix(h21(i),h21(i+vec2(1,0)),f.x),mix(h21(i+vec2(0,1)),h21(i+vec2(1,1)),f.x),f.y);
}
float n3(vec3 p){
  vec3 i=floor(p),f=fract(p); f=f*f*(3.0-2.0*f);
  return mix(
    mix(mix(h13(i),h13(i+vec3(1,0,0)),f.x),mix(h13(i+vec3(0,1,0)),h13(i+vec3(1,1,0)),f.x),f.y),
    mix(mix(h13(i+vec3(0,0,1)),h13(i+vec3(1,0,1)),f.x),mix(h13(i+vec3(0,1,1)),h13(i+vec3(1,1,1)),f.x),f.y),f.z);
}
float fbm2(vec2 p){ float v=0.0,a=0.5,fr=1.0; for(int i=0;i<5;i++){v+=a*n2(p*fr);a*=0.5;fr*=2.07;} return v; }
float fbm3(vec3 p){ float v=0.0,a=0.5,fr=1.0; for(int i=0;i<4;i++){v+=a*n3(p*fr);a*=0.5;fr*=2.07;} return v; }
mat3 rotX(float a){ float s=sin(a),c=cos(a); return mat3(1,0,0,0,c,-s,0,s,c); }

// ════════════════════════════════════════════════════════
//  BLACKBODY  800 K → 55 000 K
// ════════════════════════════════════════════════════════
vec3 blackbody(float T){
  T=clamp(T,800.0,55000.0);
  float r=T<6600.0?1.0:clamp(1.29*pow(T/6600.0,-0.133),0.0,1.0);
  float g=T<6600.0?clamp(-0.46+0.44*log(T/1000.0),0.0,1.0):clamp(1.13*pow(T/6600.0,-0.076),0.0,1.0);
  float b=T<2000.0?0.0:T>7000.0?1.0:clamp(-1.38+0.55*log(T/1000.0),0.0,1.0);
  return vec3(r,g,b);
}

// ════════════════════════════════════════════════════════
//  STARFIELD  — 3-D direction-based, no UV distortion
// ════════════════════════════════════════════════════════
vec3 starField(vec3 dir){
  dir=normalize(dir);
  float lat=abs(dir.y);
  vec3 col=mix(vec3(0.003,0.002,0.008), vec3(0.0014,0.0017,0.0032), uObservatoryMode);

  float moodBlue    = smoothstep(0.8,1.8,uPresetMood);
  float moodInferno = smoothstep(1.8,2.8,uPresetMood)*(1.0-smoothstep(2.8,3.8,uPresetMood));
  col += vec3(0.009,0.005,0.020)*(1.0-pow(lat,0.7));

  float band=exp(-dir.y*dir.y*3.0);
  float mw  =fbm3(dir*4.1+vec3(0.8,0.2,0.9));
  float mw2 =fbm3(dir*7.5+vec3(2.0,1.2,0.4));
  vec3 mwColor=mix(vec3(0.075,0.058,0.175),vec3(0.030,0.080,0.210),moodBlue);
  mwColor=mix(mwColor,vec3(0.145,0.045,0.020),moodInferno);
  col += mwColor*band*mw;
  col += vec3(0.040,0.065,0.190)*exp(-dir.y*dir.y*7.5)*0.45;
  col += vec3(0.028,0.010,0.060)*band*mw2*0.50;

  float nb1=fbm3(dir*3.0+vec3(1.3,0.5,2.1)), nb2=fbm3(dir*2.2+vec3(3.2,1.3,0.8));
  col += vec3(0.034,0.008,0.078)*nb1*nb1*exp(-lat*2.3);
  col += vec3(0.007,0.025,0.080)*nb2*nb2*exp(-lat*3.2);

  float sc[4]; sc[0]=19.0; sc[1]=54.0; sc[2]=118.0; sc[3]=242.0;
  float th[4]; th[0]=0.9960; th[1]=0.9966; th[2]=0.9973; th[3]=0.9979;
  for(int i=0;i<4;i++){
    vec3 p=floor(dir*sc[i]); float hv=h13(p);
    if(hv>th[i]){
      float br=pow((hv-th[i])/(1.0-th[i]),1.15)*(2.3+float(i)*0.45);
      float tc=h13(p*5.2);
      float tw=0.86+0.14*sin(uTime*1.5+h13(p*3.8)*21.0);
      vec3 c=tc>0.74?vec3(0.38,0.52,1.00):tc>0.52?vec3(0.82,0.93,1.00):tc>0.30?vec3(1.00,0.95,0.70):vec3(1.00,0.35,0.09);
      col+=c*br*tw;
    }
  }
  // Hero stars with diffraction spikes
  vec3 ps=floor(dir*58.0); float hs=h13(ps);
  if(hs>0.989){
    float br=pow((hs-0.989)/0.011,1.0)*5.2;
    vec2 lc=fract(dir.xy*58.0)-0.5;
    float glow=exp(-dot(lc,lc)*14.0);
    float spkH=exp(-lc.y*lc.y*120.0)*exp(-abs(lc.x)*2.8);
    float spkV=exp(-lc.x*lc.x*120.0)*exp(-abs(lc.y)*2.8);
    col+=vec3(0.88,0.94,1.00)*br*(glow+0.26*(spkH+spkV));
  }
  col = desat(col, 0.10 * uObservatoryMode);
  return col;
}

// ════════════════════════════════════════════════════════
//  HOT SPOT  — orbiting plasma blob (like M87*, Sgr A*)
//
//  Real VLBI imaging (EHT 2019, 2022) shows bright compact
//  blobs orbiting near the ISCO.  This adds 3 such spots at
//  different radii with Keplerian angular velocities.
//  They contribute ONLY through diskColor, not separately,
//  so they automatically get Doppler shift and beaming.
// ════════════════════════════════════════════════════════
float hotSpot(vec2 xz, float orbitR, float angVel, float seed){
  float rho   = length(xz);
  float rDelta = abs(rho - orbitR);
  if(rDelta > 0.7) return 0.0;
  float phi      = atan(xz.y, xz.x);
  float spotAngle = mod(uTime*angVel + seed*TAU, TAU);
  float dPhi      = mod(phi - spotAngle + PI, TAU) - PI;  // wrap to [-PI,PI]
  return exp(-dPhi*dPhi*20.0) * exp(-rDelta*rDelta*32.0);
}

// ════════════════════════════════════════════════════════
//  ACCRETION DISK
//
//  BRIGHTNESS FIXES vs previous version:
//  · col *= intens*1.45  (was *2.45  — was blowing out highlights)
//  · corona  *0.72       (was *1.20)
//  · filament *0.24      (was *0.42)
//  · far-side shadow 0.18 (was 0.34 — makes dark side genuinely dark)
//  · removed pre-ACES compression from main() — ACES alone is cleaner
//
//  NEW ADDITIONS:
//  · 3 orbiting hot spots (Keplerian, automatically Doppler-shifted)
//  · Magnetic reconnection flares in inner disk
//  · ISCO edge shock — narrow very hot ring at inner edge
// ════════════════════════════════════════════════════════
vec4 shadeDisk(vec3 worldPos, vec3 rd){
  mat3 toDisk=rotX(-uDiskTilt), fromDisk=rotX(uDiskTilt);
  vec3 p=toDisk*worldPos;
  float r=length(p.xz);
  if(r<DISK_IN||r>DISK_OUT) return vec4(0.0);

  float rn=(r-DISK_IN)/(DISK_OUT-DISK_IN);
  float phi=atan(p.z,p.x);

  vec3 orbLocal=normalize(vec3(-p.z,0.0,p.x));
  vec3 orbWorld=normalize(fromDisk*orbLocal);
  vec3 photonDir=normalize(-rd);

  // Keplerian β, capped at 0.84
  float beta=min(0.84,sqrt(0.50/max(r,DISK_IN)));

  // Full relativistic Doppler  δ = √(1-β²) / (1 - β cosθ)
  float cosT=dot(photonDir,orbWorld);
  float dop=sqrt(1.0-beta*beta)/max(1.0-beta*cosT,0.04);
  dop=clamp(dop,0.08,12.0);

  // Temperature profile
  float Tdisk=36000.0*pow(DISK_IN/max(r,DISK_IN*1.001),0.75);
  float gz=sqrt(max(0.0,1.0-1.0/max(r,HORIZON+0.05)));
  float Tobs=clamp(Tdisk*dop*gz,800.0,55000.0);
  vec3  col=blackbody(Tobs);

  // ── Density structure ───────────────────────────────
  float rphi =phi-uTime*0.33;
  float arm1 =0.5+0.5*sin(r*2.3-2.0*rphi+sin(r*0.65)*2.9);
  float arm2 =0.5+0.5*sin(r*3.6+2.0*rphi-uTime*0.27);
  float spiral=pow(max(mix(arm1,arm2,0.36),0.0),0.42);
  float turb  =0.42+0.58*fbm2(p.xz*9.0+vec2(uTime*0.22,-uTime*0.15));
  float rings =0.48+0.52*sin(r*36.0-uTime*2.9+turb*4.2);
  float innerG=exp(-rn*3.6);
  float ifade =smoothstep(0.0,0.055,rn);
  float ofade =1.0-smoothstep(0.70,1.0,rn);
  float radial=innerG*ifade*ofade;
  float dens  =radial*spiral*turb*mix(0.5,1.55,rings);
  float clump = 0.88 + 0.16*fbm2(p.xz*1.8 + vec2(4.3,-1.7));
  float dustLanes = 1.0 - pow(sat(0.62 - fbm2(p.xz*3.2 + vec2(uTime*0.03,-uTime*0.02))), 2.0) * 0.18 * uObservatoryMode;
  dens *= clump * dustLanes;

  // Disk Traverse Mode: extra close-range plasma structure so the disk feels like a place, not just a sheet.
  vec3 camDisk = toDisk * uCamPos;
  float nearTraverse = uTraverseMode * (1.0 - smoothstep(0.12, 0.72, abs(camDisk.y)));
  float micro = fbm2(p.xz * 34.0 + vec2(uTime * 0.95, -uTime * 0.70));
  float ribbons = 0.5 + 0.5 * sin(phi * 28.0 + r * 18.0 - uTime * 5.5 + micro * 6.0);
  float folds = 0.5 + 0.5 * sin(phi * 16.0 - r * 10.0 + uTime * 3.2);
  float cloudBands = fbm2(p.xz * (2.2 + 1.4 * uCloudDensity) + vec2(uTime * 0.09, -uTime * 0.06));
  float anvil = fbm2(vec2(phi * 4.2, r * 0.85 - uTime * 0.05));
  float cloudMask = smoothstep(0.36, 0.74, mix(cloudBands, anvil, 0.42));
  dens *= mix(1.0, 1.0 + (0.55 + 0.55 * uCloudDensity) * (0.35 + 0.65 * micro) * (0.35 + 0.65 * ribbons), nearTraverse);
  dens *= 1.0 + cloudMask * 0.28 * uCloudDensity;

  // Beaming  I ∝ δ^3.35
  float beam  = pow(dop,3.35);
  float intens= dens*beam*uDiskPower;

  // ── FIX: was *2.45, caused highlight blowout ────────
  col *= intens * 1.45;
  col = desat(col, 0.08 * uObservatoryMode);

  // Magnetic shear filaments
  float shear=sin(phi*9.0+r*5.0-uTime*2.5+turb*3.5);
  float magFilament=pow(0.5+0.5*shear,8.0);
  vec3 filColor=mix(vec3(1.0,0.22,0.05),vec3(0.30,0.62,1.0),smoothstep(0.75,1.7,dop));
  // FIX: was *0.42, now *0.24
  col += filColor*magFilament*radial*beam*0.24*uDiskPower;

  // Traverse glow ribbons / hot channels visible only at low altitude
  float channel = pow(ribbons, 7.0) * (0.35 + 0.65 * folds) * nearTraverse;
  col += mix(vec3(1.0, 0.44, 0.12), vec3(1.0, 0.92, 0.72), smoothstep(0.9, 2.0, dop)) * channel * beam * 0.26;

  // Cloud crests / glowing storm fronts for a more cinematic disk texture
  float crest = pow(smoothstep(0.54, 0.92, cloudMask) * (0.55 + 0.45 * micro), 1.5);
  col += mix(vec3(1.0, 0.34, 0.08), vec3(1.0, 0.82, 0.62), smoothstep(0.8, 1.7, dop)) * crest * nearTraverse * beam * (0.16 + 0.12 * uCloudDensity);

  // Dark cloud canyons / absorption gaps
  float canyon = smoothstep(0.58, 0.92, 1.0 - cloudBands) * nearTraverse;
  col *= 1.0 - canyon * 0.16 * uCloudDensity;

  // Inner corona (hot plasma above ISCO)
  float rim   = exp(-(r-DISK_IN)*7.5);
  // FIX: was *1.20, now *0.72
  col += blackbody(clamp(Tobs*1.65,800.0,55000.0))*rim*0.72*beam*uDiskPower;

  // Blue-white ergosphere edge glow
  col += vec3(0.22,0.46,0.95)*exp(-(r-DISK_IN)*5.0)*radial*beam*0.28;

  // Warm outer haze
  col += blackbody(4200.0)*rn*rn*(0.018+0.025*turb)*ifade*0.45;

  // Subtle dark absorption lanes and asymmetry for a less synthetic look
  float lane = smoothstep(0.12, 0.80, rn) * (1.0 - smoothstep(0.82, 1.0, rn));
  col *= 1.0 - lane * (0.08 + 0.10 * fbm2(p.xz*4.8 + vec2(1.2,3.4))) * uObservatoryMode;

  // ── ISCO edge shock  — sharp hot ring at inner edge ─
  // Plasma shocks as it crosses the ISCO: suddenly heats to ~50 000 K
  float iscoShock = exp(-(r-DISK_IN)*22.0) * ifade;
  col += blackbody(50000.0)*iscoShock*beam*0.55*uDiskPower;

  // ── Orbiting hot spots (M87*-style plasma blobs) ────
  // Keplerian angular velocities: ω = √(0.5/r³)
  // At r=1.38: ω≈1.18,  r=2.10: ω≈0.52,  r=3.20: ω≈0.31
  float hs1 = hotSpot(p.xz, DISK_IN*1.25, 1.18, 0.31);
  float hs2 = hotSpot(p.xz, DISK_IN*1.90, 0.52, 0.67);
  float hs3 = hotSpot(p.xz, DISK_IN*2.90, 0.31, 0.14);
  float hsAll = max(max(hs1,hs2),hs3);
  if(hsAll > 0.005){
    // Hot spots are ~2x hotter than local disk, strongly beamed
    float HsT = clamp(Tobs*(1.6+hsAll*1.8), 800.0, 55000.0);
    col += blackbody(HsT) * hsAll * beam * 1.45 * uDiskPower;
  }

  // ── Magnetic reconnection flares (inner disk only) ──
  // Stochastic: flare erupts, peaks, fades over ~1.5 time units
  float flareSeed = h11(floor(r*4.5+floor(phi/1.2)));
  float flarePhase= mod(uTime*0.18 + flareSeed, 1.0);
  float flareEnv  = smoothstep(0.0,0.20,flarePhase)*smoothstep(0.90,0.35,flarePhase);
  float flareAzim = 0.5+0.5*sin(phi*4.0+r*3.0+uTime*1.5);
  float flare     = flareEnv*pow(flareAzim,3.0)*exp(-(r-DISK_IN)*5.0)*ifade;
  col += blackbody(52000.0)*flare*beam*radial*0.75*uDiskPower;

  // ── Far-side self-shadowing (improved geometry) ─────
  // A disk point is "behind" the BH if it's on the opposite
  // side from the camera AND within the shadow cone.
  // use: shadow when  dot(diskPos, camDir) < 0  AND  r small
  vec3 camDir  = normalize(uCamPos);
  float behind = -dot(normalize(worldPos), camDir);               // >0 if behind BH
  float cone   = smoothstep(3.8, 1.3, r) * smoothstep(-0.25,0.55,behind);
  // FIX: was mixing to 0.34 (still fairly bright), now 0.16 (genuinely dark)
  col *= mix(1.0, 0.16, cone*uShadowPower);

  float alpha=ifade*ofade*clamp(0.42+intens*0.40,0.0,0.92);
  return vec4(col,alpha);
}

// ════════════════════════════════════════════════════════
//  POLAR JETS — improved with plasma knots
// ════════════════════════════════════════════════════════
vec3 polarJet(vec3 ro, vec3 rd){
  vec3 col=vec3(0.0); float t=0.0;
  for(int i=0;i<56;i++){
    vec3  p       = ro+rd*t;
    float y       = abs(p.y);
    float rJet    = length(p.xz);

    // Parabolic jet profile: wider at base, collimates upward
    float jetRadius = 0.055 + y*0.038 + y*y*0.0012;
    float core  = exp(-rJet*rJet/max(jetRadius*jetRadius,0.0001));
    float axial = smoothstep(0.85,2.0,y)*(1.0-smoothstep(8.5,15.0,y));
    float turb  = fbm3(p*2.1+vec3(0.0,uTime*0.70,0.0));

    // Plasma knots: discrete blobs moving along jet axis
    // Two knot trains at different speeds (inner + outer jet)
    float k1Period=1.8, k2Period=2.8;
    float k1Phase=fract(y/k1Period - uTime*0.42 + h11(floor(y/k1Period))*0.5);
    float k2Phase=fract(y/k2Period - uTime*0.25 + h11(floor(y/k2Period+7.3))*0.5);
    float knot1=exp(-abs(k1Phase-0.5)*18.0);
    float knot2=exp(-abs(k2Phase-0.5)*12.0)*0.55;
    float knots=max(knot1,knot2);

    // Pulse (helical instability)
    float pulse=0.55+0.45*sin(y*4.8-uTime*3.6+turb*5.0);

    // Sheath (outer boundary)
    float sheath=exp(-abs(rJet-jetRadius*0.82)*16.0);

    // Colors: inner spine is white-blue, sheath is deep blue
    vec3 spineCol = mix(vec3(0.65,0.88,1.0), vec3(1.0,0.95,0.85), knots*0.4);
    vec3 sheathCol= vec3(0.04,0.18,1.0);

    col += spineCol*core*(1.0+knots*1.2)*axial*pulse*0.038;
    col += sheathCol*sheath*axial*(turb*0.5+0.5)*0.020;

    t += 0.18;
  }
  return col*uJetPower;
}

// ════════════════════════════════════════════════════════
//  RAY TRACE  — Schwarzschild geodesic + Lense-Thirring
// ════════════════════════════════════════════════════════
vec3 traceScene(vec3 ro, vec3 rd, float lensMul){
  vec3  pos      = ro;
  float totDist  = 0.0;
  float minR     = 1000.0;
  vec3  diskAccum = vec3(0.0);
  float diskAlpha = 0.0;
  vec3  hazeAccum = vec3(0.0);
  float hazeAlpha = 0.0;
  bool  swallowed = false;

  mat3 toDisk=rotX(-uDiskTilt);
  vec3 prevPos=pos; float prevY=(toDisk*pos).y;

  for(int i=0;i<MAX_STEPS;i++){
    if(float(i)>uQualitySteps) break;
    float r=length(pos); minR=min(minR,r);

    if(r<HORIZON*0.88){ swallowed=true; break; }
    if(totDist>FAR_DIST) break;

    float dt;
    if      (r<1.05) dt=0.016;
    else if (r<1.60) dt=0.030;   // tight near photon sphere
    else if (r<3.20) dt=0.075;
    else if (r<8.00) dt=0.175;
    else             dt=0.380;

    prevPos=pos; pos+=rd*dt; totDist+=dt;

    // Schwarzschild null geodesic  acc = -(3/2)·|h|²/r⁵ · pos
    float r2=dot(pos,pos), rr=sqrt(r2), r4=r2*r2;
    vec3  hv=cross(pos,rd); float h2=dot(hv,hv);
    vec3  acc=-(1.5*h2/(r4*rr))*pos*(uLensStrength*lensMul);

    // Kerr frame drag (Lense-Thirring)
    float a_M=uSpinWarp*0.5;
    vec3  Sv=vec3(0.0,a_M,0.0); float Sdr=dot(Sv,pos)/rr;
    vec3  Bgm=(3.0*Sdr*pos/r2-Sv)/(r2*rr);
    acc+=2.0*1.5*cross(rd,Bgm);

    rd=normalize(rd+acc*dt);

    // Volumetric accretion haze — gives the disk believable thickness
    vec3 hazeP = toDisk * pos;
    float hazeR = length(hazeP.xz);
    if(hazeR > DISK_IN*0.95 && hazeR < DISK_OUT*1.03){
      float thickness = mix(0.028, 0.12, smoothstep(DISK_IN, DISK_OUT, hazeR));
      thickness *= mix(1.0, 1.75, uTraverseMode);
      float slab = exp(-abs(hazeP.y) / max(thickness, 0.0001));
      float shell = smoothstep(DISK_IN*0.98, DISK_IN*1.42, hazeR) * (1.0 - smoothstep(DISK_OUT*0.70, DISK_OUT*1.03, hazeR));
      float noiseH = 0.55 + 0.45 * fbm2(hazeP.xz*1.9 + vec2(uTime*0.05, -uTime*0.03));
      float cloudLarge = fbm2(hazeP.xz * (1.0 + 0.8 * uCloudDensity) + vec2(uTime * 0.03, -uTime * 0.015));
      float cloudFine = fbm2(hazeP.xz * 5.2 + vec2(-uTime * 0.16, uTime * 0.08));
      float weather = smoothstep(0.34, 0.84, mix(cloudLarge, cloudFine, 0.35));
      float loft = 0.65 + 0.35 * sin(hazeP.y * 34.0 + cloudFine * 6.0);
      float cloudD = slab * shell * noiseH * (0.55 + 0.80 * weather * uCloudDensity) * loft;
      float hazeD = cloudD * uDiskHaze * mix(0.028, 0.052, uTraverseMode);
      vec3 hazeCol = mix(blackbody(3200.0), blackbody(12000.0), exp(-(hazeR-DISK_IN)*0.58));
      hazeCol = mix(hazeCol, vec3(1.0, 0.48, 0.16), 0.16 * weather);
      hazeCol += vec3(1.0, 0.38, 0.10) * pow(noiseH, 4.0) * 0.12 * uTraverseMode;
      hazeCol += vec3(1.0, 0.74, 0.52) * pow(weather, 3.0) * 0.08 * uCloudDensity;
      hazeAccum += hazeCol * hazeD * dt * (1.0 - hazeAlpha);
      hazeAlpha += hazeD * dt * (1.0 - hazeAlpha);
    }

    // Disk crossing
    float diskY=(toDisk*pos).y;
    if(sign(prevY)!=sign(diskY)&&diskAlpha<0.98){
      float t2=prevY/(prevY-diskY);
      vec3 hit=mix(prevPos,pos,clamp(t2,0.0,1.0));
      vec4 dc=shadeDisk(hit,rd);
      if(dc.a>0.001){ diskAccum+=dc.rgb*dc.a*(1.0-diskAlpha); diskAlpha+=dc.a*(1.0-diskAlpha); }
    }
    prevY=diskY;
  }

  if(swallowed){
    // Deep shadow — only a tiny rim glow
    float rim=exp(-(minR-HORIZON*0.9)*16.0);
    // FIX: was diskAccum*0.045 — was lifting shadow grey.  Now much darker.
    return diskAccum*0.018 + vec3(0.005,0.002,0.010)*rim;
  }

  vec3 bg=starField(rd);
  bg+=polarJet(ro,rd);

  // ── Photon rings ──────────────────────────────────────
  // FIX: ring1 was *22.0 sharpness → now *34.0 (razor thin)
  //      amplitude *1.25 → *2.80 (dramatically brighter)
  float ring1=exp(-abs(minR-PHOTON_R)*34.0);
  float ring2=exp(-abs(minR-PHOTON_R*1.115)*36.0);
  float ring3=exp(-abs(minR-PHOTON_R*0.935)*42.0);
  bg+=vec3(1.00,0.74,0.26)*ring1*2.15*uPhotonEcho*mix(1.0,0.72,uObservatoryMode);  // main golden ring
  bg+=vec3(0.55,0.80,1.00)*ring2*0.32*uPhotonEcho*mix(1.0,0.58,uObservatoryMode);  // outer blue echo
  bg+=vec3(1.00,0.28,0.08)*ring3*0.18*uPhotonEcho*mix(1.0,0.50,uObservatoryMode);  // inner red sub-ring

  // Secondary image hint (disk light bent around photon sphere)
  float sec=exp(-abs(minR-PHOTON_R*1.7)*3.0)*0.14*uPhotonEcho;
  bg+=vec3(0.75,0.45,0.15)*sec;

  // Penumbra glow (light scattered from disk onto shadow edge)
  float penumbra=exp(-abs(minR-HORIZON*1.6)*2.5)*0.08;
  bg+=diskAccum*penumbra;

  // Secondary lensed disk image approximation:
  // a compressed, faint duplicate of disk emission near the photon ring.
  float echoUpper = exp(-abs(minR - PHOTON_R*1.22)*9.5) * smoothstep(0.02, 0.42, abs(rd.y));
  vec3 secondary = diskAccum * echoUpper * uSecondaryImage * mix(0.20, 0.34, uObservatoryMode);
  secondary = desat(secondary, 0.18 * uObservatoryMode);
  bg += secondary;

  bg+=hazeAccum * mix(0.70, 1.10, uObservatoryMode);

  float compAlpha = clamp(diskAlpha + hazeAlpha*0.65, 0.0, 0.98);
  return mix(bg, bg*0.10 + diskAccum + hazeAccum*0.9, compAlpha);
}

// ════════════════════════════════════════════════════════
//  RENDER RAY  (handles Ultra RGB split mode)
// ════════════════════════════════════════════════════════
vec3 renderRay(vec2 uv, float lensMul){
  float distK=0.014*dot(uv,uv);
  vec2  uvD=uv*(1.0+distK);
  vec3  rd=normalize(uCamForward+uvD.x*uCamRight+uvD.y*uCamUp);
  return traceScene(uCamPos,rd,lensMul);
}

// ════════════════════════════════════════════════════════
//  MAIN
// ════════════════════════════════════════════════════════
void main(){
  vec2 uv=vUv*2.0-1.0;
  uv.x*=uResolution.x/uResolution.y;
  uv*=uFov;

  vec3 col;
  if(uUltra>0.5){
    float chroma=0.0016*smoothstep(1.2,0.05,length(uv));
    float lensMul = (uSplitMode > 0.5 && vUv.x < 0.5) ? 0.02 : 1.0;
    vec3 cR=renderRay(uv+vec2( chroma,0.0), lensMul);
    vec3 cG=renderRay(uv, lensMul);
    vec3 cB=renderRay(uv-vec2( chroma,0.0), lensMul);
    col=vec3(cR.r,cG.g,cB.b);
  } else {
    float lensMul = (uSplitMode > 0.5 && vUv.x < 0.5) ? 0.02 : 1.0;
    col=renderRay(uv, lensMul);
  }

  col*=uExposure;

  // Spectral highlight governor: compress only extreme peaks, preserving color detail.
  // This keeps Annihilation/Singularity modes from turning into a white blob.
  float peak = max(max(col.r, col.g), col.b);
  float gov  = smoothstep(1.15, 5.5, peak) * 0.58;
  col = mix(col, col / (1.0 + peak * 0.20), gov);

  // ── ACES filmic tone mapping ──────────────────────────
  // FIX: removed pre-compression   col/(1+col*0.22)  that
  //      was muddying highlights.  Pure ACES is cleaner.
  const float A=2.51,B=0.03,C=2.43,D=0.59,E=0.14;
  col=clamp((col*(A*col+B))/(col*(C*col+D)+E),0.0,1.0);

  // Gamma 2.2
  col=pow(max(col,vec3(0.0)),vec3(1.0/2.2));

  // Observatory-mode sensor response: slight desaturation + lens dirt on highlights
  if(uObservatoryMode > 0.5){
    col = desat(col, 0.08);
    float speck = fbm2(gl_FragCoord.xy / uResolution.y * 4.5 + vec2(3.7, 8.1));
    speck = pow(sat(speck - 0.60) * 2.6, 1.8);
    float lum = dot(col, vec3(0.2126,0.7152,0.0722));
    col += vec3(1.0, 0.95, 0.86) * speck * sat((lum - 0.42) * 1.8) * uLensDirt * 0.10;
  }

  // Event Horizon Lab: split-screen comparison marker.
  if(uSplitMode > 0.5){
    float line = 1.0 - smoothstep(0.0, 0.0035, abs(vUv.x - 0.5));
    vec3 divider = vec3(0.45, 0.78, 1.0) * line * 0.65;
    col += divider;
    if(vUv.x < 0.5) col *= vec3(0.72, 0.78, 0.92);
  }

  // Temporary matter-injection flare from Shift+Click / Mission Mode.
  float pulse = max(uMatterPulse, 0.0);
  if(pulse > 0.001){
    float rr = length(vUv*2.0-1.0);
    float shock = exp(-abs(rr - (0.20 + pulse*0.42)) * 18.0);
    vec3 flare = mix(vec3(1.0,0.35,0.08), vec3(0.35,0.72,1.0), pulse);
    col += flare * shock * pulse * 0.22;
  }

  // Film grain (very subtle)
  float grain=fract(sin(dot(gl_FragCoord.xy+uTime*37.0,vec2(12.9898,78.233)))*43758.5453);
  col+=(grain-0.5)*mix(0.007, 0.0035, uObservatoryMode);

  // Vignette
  vec2 cv=vUv*2.0-1.0;
  float vig=smoothstep(1.40, mix(0.18,0.24,uObservatoryMode), length(cv));
  col*=mix(mix(0.34,0.48,uObservatoryMode),1.0,vig);

  gl_FragColor=vec4(max(col,vec3(0.0)),1.0);
}
`;
