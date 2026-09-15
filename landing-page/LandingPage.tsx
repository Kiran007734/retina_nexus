import "../styles/landing.css";
import Navbar from "../components/landing-new/Navbar";
import HeroVisual from "../components/landing/HeroVisual";
import HeroContent from "../components/landing/HeroContent";
import RetinalShowcase from "../components/landing/RetinalShowcase";
import HowItWorks from '../components/landing/HowItWorks';
import ExplainableAI from '../components/landing/ExplainableAI';
import RuralFirst from '../components/landing/RuralFirst';
import Impact from '../components/landing/Impact';
import TechnologyInfrastructure from '../components/landing/TechnologyInfrastructure';
import ClinicalIntelligence from '../components/landing/ClinicalIntelligence';
import FinalCTA from '../components/landing/FinalCTA';
import Footer from '../components/landing/Footer';
import retinaAtmosphere from "../assets/images/retina-atmosphere.png";
const LandingPage = () => {
  return (
    <div className="landing-page">

      {/* Global red-violet atmosphere */}
      <div
        className="retina-atmosphere"
        style={{
          backgroundImage: `url(${retinaAtmosphere})`,
        }}
      />

      <Navbar />

      <HeroVisual />
      <HeroContent />

      <RetinalShowcase />

      <HowItWorks />

      <ExplainableAI />

      <RuralFirst />

      <Impact />

      <TechnologyInfrastructure />

      <ClinicalIntelligence />

      <FinalCTA />

      <Footer />

    </div>
  );
};

export default LandingPage;