import {
  Activity,
  ArrowRight,
  BrainCircuit,
  Check,
  ChevronUp,
  Eye,
  FileCheck2,
  Menu,
  Network,
  ScanEye,
  ShieldCheck,
  Stethoscope,
  X,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import '../styles/landing.css';

const navigation = [
  { label: 'Technology', target: 'technology' },
  { label: 'How It Works', target: 'how-it-works' },
  { label: 'Explainable AI', target: 'explainable-ai' },
  { label: 'Research', target: 'research' },
];

function scrollToSection(target: string) {
  document.getElementById(target)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

export function LandingPage() {
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    document.title = 'RETINA NEXUS | Explainable retinal screening';
    const observed = Array.from(document.querySelectorAll<HTMLElement>('[data-reveal]'));
    const observer = new IntersectionObserver(
      (entries) => entries.forEach((entry) => {
        if (entry.isIntersecting) entry.target.classList.add('is-visible', 'tech-visible', 'clinical-visible', 'final-cta-visible');
      }),
      { threshold: 0.12 },
    );
    observed.forEach((element) => observer.observe(element));
    return () => {
      observer.disconnect();
      document.title = 'RETINA NEXUS';
    };
  }, []);

  return (
    <div className="landing-page">
      <div className="retina-atmosphere rn-atmosphere" aria-hidden="true" />

      <header className="landing-navbar">
        <div className="landing-navbar-inner">
          <Link to="/" className="landing-brand" aria-label="RETINA NEXUS home">
            <span className="rn-brand-icon"><Eye size={20} /></span>
            <span><strong>RETINA</strong><small>NEXUS</small></span>
          </Link>
          <nav className="landing-nav" aria-label="Landing page navigation">
            {navigation.map((item) => (
              <button key={item.target} type="button" onClick={() => scrollToSection(item.target)}>{item.label}</button>
            ))}
            <Link className="landing-nav-cta" to="/login">Launch Platform <ArrowRight size={14} /></Link>
          </nav>
          <button
            className="rn-mobile-toggle"
            type="button"
            aria-label={menuOpen ? 'Close navigation' : 'Open navigation'}
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((open) => !open)}
          >
            {menuOpen ? <X size={21} /> : <Menu size={21} />}
          </button>
          {menuOpen && (
            <nav className="rn-mobile-menu" aria-label="Mobile landing page navigation">
              {navigation.map((item) => (
                <button key={item.target} type="button" onClick={() => { setMenuOpen(false); scrollToSection(item.target); }}>{item.label}</button>
              ))}
              <Link to="/login" onClick={() => setMenuOpen(false)}>Launch Platform <ArrowRight size={15} /></Link>
            </nav>
          )}
        </div>
      </header>

      <main>
        <section className="rn-hero" aria-labelledby="landing-title">
          <div className="rn-hero-grid" aria-hidden="true" />
          <div className="rn-hero-copy">
            <p className="hero-eyebrow"><span className="hero-eyebrow-dot" /> THE CARE INTELLIGENCE LAYER</p>
            <h1 id="landing-title">See clearly.<span>Screen early.</span>Protect sight.</h1>
            <p>Evidence-linked retinal screening that combines image quality, AI assessment, explainability, and human review in one accountable workflow.</p>
            <div className="hero-actions">
              <Link className="hero-primary-btn" to="/login">Launch Platform <ArrowRight size={17} /></Link>
              <button className="hero-secondary-btn" type="button" onClick={() => scrollToSection('how-it-works')}>Explore the workflow</button>
            </div>
            <div className="rn-boundary"><ShieldCheck size={16} /> Screening support for trained care teams. Clinical review remains essential.</div>
          </div>

          <div className="rn-retina-stage" aria-label="Abstract retinal intelligence visualization">
            <div className="rn-orbit rn-orbit-one" />
            <div className="rn-orbit rn-orbit-two" />
            <div className="rn-retina-disc">
              <div className="rn-vessel rn-vessel-a" />
              <div className="rn-vessel rn-vessel-b" />
              <div className="rn-vessel rn-vessel-c" />
              <div className="rn-optic-disc" />
              <div className="rn-scan" />
            </div>
            <div className="rn-floating-card rn-card-quality"><Check size={14} /> Quality-first intake</div>
            <div className="rn-floating-card rn-card-evidence"><ScanEye size={14} /> Evidence linked</div>
            <div className="rn-stage-status"><span className="status-dot" /> RETINAL WORKFLOW READY</div>
          </div>
        </section>

        <section id="technology" className="retinal-showcase rn-scroll-target" data-reveal>
          <div className="retinal-showcase-header">
            <span className="section-eyebrow"><span className="eyebrow-dot" /> RETINAL INTELLIGENCE</span>
            <h2>One image. <span>Multiple safety checks.</span></h2>
            <p>The platform keeps the primary classifier, anatomical evidence, explanation maps, and reliability assessment distinct—then presents them together for review.</p>
          </div>
          <div className="retinal-visual-card">
            <div className="retinal-image-placeholder">
              <div className="retinal-center">
                <span className="retinal-ring ring-one" />
                <span className="retinal-ring ring-two" />
                <span className="retinal-ring ring-three" />
                <span className="retinal-core" />
                <span className="scan-line" />
              </div>
            </div>
            <div className="visual-label label-top"><span /> IMAGE QUALITY GATE</div>
            <div className="visual-label label-right"><span /> EVIDENCE VERIFICATION</div>
            <div className="visual-label label-bottom"><span /> HUMAN REVIEW BOUNDARY</div>
            <div className="visual-status"><span className="status-dot" /> EXPLAINABLE BY DESIGN</div>
          </div>
        </section>

        <section id="how-it-works" className="rn-section rn-scroll-target" data-reveal>
          <div className="rn-section-heading">
            <span className="section-eyebrow"><span className="eyebrow-dot" /> HOW IT WORKS</span>
            <h2>A controlled path from capture to care.</h2>
            <p>Every stage has a clear responsibility, explicit failure state, and auditable output.</p>
          </div>
          <div className="rn-workflow">
            {[
              { icon: ScanEye, number: '01', title: 'Validate the image', body: 'File integrity, retinal field, focus, exposure, and gradability are checked before clinical AI.' },
              { icon: BrainCircuit, number: '02', title: 'Analyze responsibly', body: 'DR severity inference remains separate from lesion, vessel, and explanation evidence.' },
              { icon: Network, number: '03', title: 'Cross-check evidence', body: 'RetinaGuard combines only available quality, uncertainty, agreement, and distribution signals.' },
              { icon: Stethoscope, number: '04', title: 'Keep humans in control', body: 'Clinicians can approve, modify, reject, request recapture, or refer for specialist review.' },
            ].map(({ icon: Icon, number, title, body }) => (
              <article className="rn-workflow-card" key={number}>
                <div className="rn-card-top"><span>{number}</span><Icon size={21} /></div>
                <h3>{title}</h3><p>{body}</p>
              </article>
            ))}
          </div>
        </section>

        <section id="explainable-ai" className="rn-section rn-explain rn-scroll-target" data-reveal>
          <div className="rn-explain-visual" aria-hidden="true">
            <div className="rn-explain-retina"><span className="rn-attention rn-attention-a" /><span className="rn-attention rn-attention-b" /><span className="rn-explain-sweep" /></div>
            <div className="rn-evidence-chip rn-chip-one">GRAD-CAM</div>
            <div className="rn-evidence-chip rn-chip-two">LESION EVIDENCE</div>
            <div className="rn-evidence-chip rn-chip-three">VESSEL MAP</div>
          </div>
          <div className="rn-explain-copy">
            <span className="section-eyebrow"><span className="eyebrow-dot" /> EXPLAINABLE AI</span>
            <h2>Evidence, not just an answer.</h2>
            <p>RETINA NEXUS links a screening output with visual attention, lesion findings, vessel structure, uncertainty, and agreement checks. These engineering signals support scrutiny; they do not prove clinical causality.</p>
            <ul>
              <li><Check size={15} /> Class-specific attention visualization</li>
              <li><Check size={15} /> Attention-to-lesion agreement</li>
              <li><Check size={15} /> Transparent reliability factors</li>
              <li><Check size={15} /> Complete audit provenance</li>
            </ul>
          </div>
        </section>

        <section id="research" className="rn-section rn-research rn-scroll-target" data-reveal>
          <div className="rn-section-heading">
            <span className="section-eyebrow"><span className="eyebrow-dot" /> RESEARCH & GOVERNANCE</span>
            <h2>Built to be measured, challenged, and improved.</h2>
            <p>Model outputs and trust indicators remain versioned, reproducible, and explicitly separated from clinical validation claims.</p>
          </div>
          <div className="rn-research-grid">
            <article><Activity size={22} /><h3>Monitored operation</h3><p>Track model versions, quality distribution, review demand, latency, and drift flags without automatic retraining.</p></article>
            <article><FileCheck2 size={22} /><h3>Dataset governance</h3><p>Validation, leakage checks, manifests, checksums, and split provenance support reproducible research.</p></article>
            <article><ShieldCheck size={22} /><h3>Safety boundaries</h3><p>Missing signals remain missing, module failures remain visible, and no secondary verifier silently overrides the primary result.</p></article>
          </div>
        </section>

        <section className="final-cta rn-final" data-reveal>
          <div className="final-cta-violet-glow" aria-hidden="true" />
          <div className="rn-final-content">
            <span className="section-eyebrow"><span className="eyebrow-dot" /> RETINA NEXUS PLATFORM</span>
            <h2>Bring explainable retinal screening into one secure workspace.</h2>
            <p>Continue to the existing authenticated prototype for screening, evidence review, RetinaGuard, reporting, and clinician decisions.</p>
            <Link className="hero-primary-btn" to="/login">Launch Platform <ArrowRight size={17} /></Link>
          </div>
        </section>
      </main>

      <footer className="site-footer rn-footer">
        <Link to="/" className="landing-brand" aria-label="RETINA NEXUS home">
          <span className="rn-brand-icon"><Eye size={19} /></span><span><strong>RETINA</strong><small>NEXUS</small></span>
        </Link>
        <p>Evidence-linked retinal screening support. Prototype—not a medical diagnosis or regulatory approval claim.</p>
        <button type="button" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>Back to top <ChevronUp size={15} /></button>
      </footer>
    </div>
  );
}
